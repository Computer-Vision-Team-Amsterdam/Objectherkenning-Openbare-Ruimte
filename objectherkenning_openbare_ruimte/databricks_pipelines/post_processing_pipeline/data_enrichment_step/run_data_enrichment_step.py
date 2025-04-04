# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from datetime import datetime  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.detections import (  # noqa: E402
    SilverDetectionMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (  # noqa: E402
    SilverFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.objects import (  # noqa: E402
    SilverObjectsPerDayManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components import (  # noqa: E402
    utils_scoring,
    utils_visualization,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.clustering_detections import (  # noqa: E402
    Clustering,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.decos_data_connector import (  # noqa: E402
    DecosDataHandler,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.vulnerable_bridges_handler import (  # noqa: E402
    VulnerableBridgesHandler,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_data_enrichment_step(
    sparkSession,
    catalog,
    schema,
    root_source,
    vuln_bridges_relative_path,
    az_tenant_id,
    db_host,
    db_name,
    device_id,
    job_process_time,
    active_object_classes,
    permit_mapping,
    confidence_thresholds,
    bbox_size_thresholds,
    annotate_detection_images,
):
    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)
    clustering = Clustering(
        spark=sparkSession,
        catalog=catalog,
        schema=schema,
        detections=SilverDetectionMetadataManager.load_pending_rows_from_table(),
        frames=SilverFrameMetadataManager.load_pending_rows_from_table(),
        active_object_classes=active_object_classes,
        confidence_thresholds=confidence_thresholds,
        bbox_size_thresholds=bbox_size_thresholds,
    )
    objects_coordinates_df = clustering.get_objects_coordinates_with_detection_id()
    category_counts = sorted(
        objects_coordinates_df.groupBy("object_class").count().collect()
    )
    for row in category_counts:
        print(
            f"Detected '{active_object_classes[row['object_class']]}': {row['count']}"
        )

    bridgesHandler = VulnerableBridgesHandler(
        spark=sparkSession,
        root_source=root_source,
        vuln_bridges_relative_path=vuln_bridges_relative_path,
    )
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()
    closest_bridges_df = (
        bridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
            bridges_locations_as_linestrings=bridges_coordinates_geometry,
            objects_coordinates_df=objects_coordinates_df,
            bridges_ids=bridgesHandler.get_bridges_ids(),
            bridges_coordinates=bridgesHandler.get_bridges_coordinates(),
        )
    )
    objects_coordinates_with_closest_bridge_df = objects_coordinates_df.join(
        closest_bridges_df, "detection_id"
    )

    decosDataHandler = DecosDataHandler(
        spark=sparkSession,
        az_tenant_id=az_tenant_id,
        db_host=db_host,
        db_name=db_name,
        db_port=5432,
        active_object_classes=active_object_classes,
        permit_mapping=permit_mapping,
    )
    decosDataHandler.query_and_process_object_permits(
        date_to_query=datetime.today().strftime("%Y-%m-%d")
    )
    closest_permits_df = decosDataHandler.calculate_distances_to_closest_permits(
        objects_coordinates_df=objects_coordinates_df,
    )
    objects_coordinates_with_closest_bridge_and_closest_permit_df = (
        objects_coordinates_with_closest_bridge_df.join(
            closest_permits_df, "detection_id"
        )
    )

    score_expr = utils_scoring.get_score_expr()
    objects_coordinates_with_closest_bridge_and_closest_permit_and_score_df = (
        objects_coordinates_with_closest_bridge_and_closest_permit_df.withColumn(
            "score", score_expr
        )
    )

    joined_metadata_with_closest_bridge_and_closest_permit_and_score_df = (
        objects_coordinates_with_closest_bridge_and_closest_permit_and_score_df.alias(
            "a"
        ).join(
            clustering.joined_metadata.alias("b"),
            on=F.col("a.detection_id") == F.col("b.detection_id"),
        )
    )

    utils_visualization.generate_map(
        dataframe=joined_metadata_with_closest_bridge_and_closest_permit_and_score_df,
        annotate_detection_images=annotate_detection_images,
        name=f"{job_process_time}-map",
        path=f"/Volumes/{catalog}/default/landingzone/Luna/visualizations/{datetime.today().strftime('%Y-%m-%d')}/",
        catalog=catalog,
        device_id=device_id,
        job_process_time=job_process_time,
    )

    selected_casted_df = (
        joined_metadata_with_closest_bridge_and_closest_permit_and_score_df.select(
            F.col("a.detection_id").cast("int"),
            F.col("a.object_class"),
            F.col("b.gps_lat").alias("object_lat").cast("string"),
            F.col("b.gps_lon").alias("object_lon").cast("string"),
            F.col("closest_bridge_distance")
            .alias("distance_closest_bridge")
            .cast("float"),
            F.col("closest_bridge_id").cast("string"),
            F.col("closest_permit_distance")
            .alias("distance_closest_permit")
            .cast("float"),
            F.col("closest_permit_id"),
            F.col("closest_permit_lat").cast("float"),
            F.col("closest_permit_lon").cast("float"),
            F.col("score").cast("float"),
            F.lit("Pending").alias("status"),
        )
    )

    SilverObjectsPerDayManager.insert_data(df=selected_casted_df)
    SilverFrameMetadataManager.update_status(job_process_time=job_process_time)
    SilverDetectionMetadataManager.update_status(job_process_time=job_process_time)


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataEnrichment").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_data_enrichment_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        root_source=settings["storage_account_root_path"],
        vuln_bridges_relative_path=settings["vuln_bridges_relative_path"],
        az_tenant_id=settings["azure_tenant_id"],
        db_host=settings["reference_database"]["host"],
        db_name=settings["reference_database"]["name"],
        device_id=settings["device_id"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
        active_object_classes=settings["object_classes"]["active"],
        permit_mapping=settings["object_classes"]["permit_mapping"],
        confidence_thresholds=settings["object_classes"]["confidence_threshold"],
        bbox_size_thresholds=settings["object_classes"]["bbox_size_threshold"],
        annotate_detection_images=settings["annotate_detection_images"],
    )
