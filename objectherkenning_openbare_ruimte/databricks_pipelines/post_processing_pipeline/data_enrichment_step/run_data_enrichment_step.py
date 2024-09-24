# flake8: noqa
# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from datetime import datetime  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import FloatType  # noqa: E402

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
    job_process_time,
):
    # Setup clustering
    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)

    clustering = Clustering(
        spark=sparkSession,
        catalog=catalog,
        schema=schema,
        detections=SilverDetectionMetadataManager.load_pending_rows_from_table(),
        frames=SilverFrameMetadataManager.load_pending_rows_from_table(),
    )
    clustering.setup()
    containers_coordinates_df = (
        clustering.get_containers_coordinates_with_detection_id()
    )

    # Setup bridges data
    bridgesHandler = VulnerableBridgesHandler(
        spark=sparkSession,
        root_source=root_source,
        vuln_bridges_relative_path=vuln_bridges_relative_path,
    )
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()

    # Setup permit data
    decosDataHandler = DecosDataHandler(
        spark=sparkSession,
        az_tenant_id=az_tenant_id,
        db_host=db_host,
        db_name=db_name,
        db_port=5432,
    )

    print(f"03: Number of containers: {containers_coordinates_df.count()}.")

    # Enrich with bridges data
    closest_bridges_df = (
        bridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
            bridges_locations_as_linestrings=bridges_coordinates_geometry,
            containers_coordinates_df=containers_coordinates_df,
            bridges_ids=bridgesHandler.get_bridges_ids(),
            bridges_coordinates=bridgesHandler.get_bridges_coordinates(),
        )
    )

    containers_coordinates_with_closest_bridge_df = containers_coordinates_df.join(
        closest_bridges_df, "detection_id"
    )

    # Enrich with decos data
    # date_to_query = datetime.today().strftime("%Y-%m-%d")
    # date_to_query = job_process_time.strftime("%Y-%m-%d")
    date_to_query = "2024-08-27"
    query = f"SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"  # nosec B608
    print(f"Querying the database for date {date_to_query}...")
    decosDataHandler.run(query)
    decosDataHandler.process_query_result()

    closest_permits_df = decosDataHandler.calculate_distances_to_closest_permits(
        permits_locations_as_points=decosDataHandler.get_permits_coordinates_geometry(),
        permits_ids=decosDataHandler.get_permits_ids(),
        permits_coordinates=decosDataHandler.get_permits_coordinates(),
        containers_coordinates_df=containers_coordinates_df,
    )

    containers_coordinates_with_closest_bridge_and_closest_permit_df = (
        containers_coordinates_with_closest_bridge_df.join(
            closest_permits_df, "detection_id"
        )
    )

    # Enrich with score
    calculate_score_spark_udf = F.udf(calculate_score, FloatType())
    containers_coordinates_with_closest_bridge_and_closest_permit_and_score_df = containers_coordinates_with_closest_bridge_and_closest_permit_df.withColumn(
        "score",
        calculate_score_spark_udf(
            containers_coordinates_with_closest_bridge_and_closest_permit_df.closest_bridge_distance,
            containers_coordinates_with_closest_bridge_and_closest_permit_df.closest_permit_distance,
        ),
    )

    display(clustering.joined_metadata)
    display(containers_coordinates_with_closest_bridge_and_closest_permit_and_score_df)

    # df_joined_with_closest_bridge_and_closest_permit_and_score_df = (
    #     clustering.df_joined.join(
    #         containers_coordinates_with_closest_bridge_and_closest_permit_and_score_df,
    #         "detection_id",
    #     ).drop(clustering.df_joined)
    # )
    # joined_metadata_with_closest_bridge_and_closest_permit_and_score_df = (
    #     containers_coordinates_with_closest_bridge_and_closest_permit_and_score_df.join(
    #         clustering.joined_metadata.withColumnRenamed(
    #             "gps_lat", "gps_lat_copy"
    #         ).withColumnRenamed("gps_lon", "gps_lon_copy"),
    #         on="detection_id",
    #     )
    # )

    # print("Before drop")
    # display(joined_metadata_with_closest_bridge_and_closest_permit_and_score_df)
    # joined_metadata_with_closest_bridge_and_closest_permit_and_score_df = (
    #     joined_metadata_with_closest_bridge_and_closest_permit_and_score_df.drop(
    #         "gps_lat_copy", "gps_lon_copy"
    #     )
    # )
    # print("After drop")
    # display(joined_metadata_with_closest_bridge_and_closest_permit_and_score_df)

    clustering.frame_metadata = clustering.frame_metadata.drop("gps_lat", "gps_lon")
    print("dropped gps from joined metadata")
    display(clustering.frame_metadata)
    joined_metadata_with_closest_bridge_and_closest_permit_and_score_df = containers_coordinates_with_closest_bridge_and_closest_permit_and_score_df.alias(
        "a"
    ).join(
        clustering.joined_metadata.alias("b"),
        on=F.col("a.detection_id") == F.col("b.detection_id"),
    )

    print("After join")
    display(joined_metadata_with_closest_bridge_and_closest_permit_and_score_df)
    # # Gather data to visualize
    utils_visualization.generate_map(
        dataframe=joined_metadata_with_closest_bridge_and_closest_permit_and_score_df,
        name=f"{job_process_time}-map",
        path=f"/Volumes/{catalog}/default/landingzone/Luna/visualizations/{date_to_query}/",
    )

    selected_casted_df = (
        joined_metadata_with_closest_bridge_and_closest_permit_and_score_df.select(
            F.col("a.detection_id").cast("int"),
            F.col("object_class"),
            F.col("a.gps_lat").alias("object_lat").cast("string"),
            F.col("a.gps_lon").alias("object_lon").cast("string"),
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

    print("Final df")
    display(selected_casted_df)

    # SilverObjectsPerDayManager.insert_data(df=selected_casted_df)
    # SilverFrameMetadataManager.update_status(job_process_time=job_process_time)
    # SilverDetectionMetadataManager.update_status(job_process_time=job_process_time)


def calculate_score(bridge_distance: float, permit_distance: float) -> float:
    """
    Calculate score for bridge and permit distance;
    """
    if permit_distance >= 40 and bridge_distance < 25:
        return 1 + max([(25 - bridge_distance) / 25, 0])
    elif permit_distance >= 40 and bridge_distance >= 25:
        return min(1.0, permit_distance / 100.0)
    else:
        return 0


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataEnrichment").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_db.yml")
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
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
    )
