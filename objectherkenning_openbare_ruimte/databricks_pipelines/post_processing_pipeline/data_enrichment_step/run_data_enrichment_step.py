# Run clustering
# enrich with Decos data and with Bridges data
# prioritize based on score
# store results in tables.

# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from datetime import datetime  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.table_manager import (  # noqa: E402
    TableManager,
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
    clustering = Clustering(spark=sparkSession, catalog=catalog, schema=schema)
    clustering.setup()
    containers_coordinates_geometry = clustering.get_containers_coordinates_geometry()

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

    tableManager = TableManager(spark=sparkSession, catalog=catalog, schema=schema)

    print(
        f"03: Number of containers: {len(containers_coordinates_geometry)}."
    )
    # Enrich with bridges data
    (
        closest_bridges_distances,
        closest_bridges_ids,
        closest_bridges_coordinates,
        closest_bridges_wkts,
    ) = VulnerableBridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
        bridges_locations_as_linestrings=bridges_coordinates_geometry,
        containers_locations_as_points=containers_coordinates_geometry,
        bridges_ids=bridgesHandler.get_bridges_ids(),
        bridges_coordinates=bridgesHandler.get_bridges_coordinates(),
    )

    clustering.add_columns({
        "closest_bridge_distance": closest_bridges_distances,
        "closest_bridge_id": closest_bridges_ids,
        "closest_bridge_coordinates": closest_bridges_coordinates,
        "closest_bridge_linestring_wkt": closest_bridges_wkts
    })

    # Enrich with decos data
    date_to_query = datetime.today().strftime("%Y-%m-%d")
    query = f"SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"  # nosec B608
    print(f"Querying the database for date {date_to_query}...")
    decosDataHandler.run(query)
    decosDataHandler.process_query_result()

    permit_distances, closest_permits, closest_permits_coordinates = (
        decosDataHandler.calculate_distances_to_closest_permits(
            permits_locations_as_points=decosDataHandler.get_permits_coordinates_geometry(),
            permits_ids=decosDataHandler.get_permits_ids(),
            permits_coordinates=decosDataHandler.get_permits_coordinates(),
            containers_locations_as_points=containers_coordinates_geometry,
        )
    )

    clustering.add_columns({
        "closest_permit_distance": permit_distances,
        "closest_permit_id": closest_permits,
        "closest_permit_coordinates": closest_permits_coordinates
    })

    # Enrich with score
    scores = [
        float(calculate_score(closest_bridges_distances[idx], permit_distances[idx]))
        for idx in range(len(clustering.get_containers_coordinates()))
    ]
    clustering.add_column(column_name="score", values=scores)

    # Gather data to visualize
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{current_datetime}-map"
    path = f"/Volumes/{clustering.catalog}/default/landingzone/Luna/visualizations/{date_to_query}/"

    utils_visualization.generate_map(
        dataframe=clustering.df_joined,
        name=name,
        path=path,
    )

    clustering.df_joined = (
    clustering.df_joined
    .select(
        col("detection_id"),
        col("object_class"),
        col("gps_lat").alias("object_lat"),
        col("gps_lon").alias("object_lon"),
        col("closest_bridge_distance").alias("distance_closest_bridge"),
        col("closest_bridge_id"),
        col("closest_permit_distance").alias("distance_closest_permit"),
        col("closest_permit_id"),
        col("closest_permit_coordinates"),
        col("score")
    )
)

    clustering.df_joined = (clustering.df_joined.withColumn("closest_permit_lat", F.col("closest_permit_coordinates._1"))
        .withColumn("closest_permit_lon", F.col("closest_permit_coordinates._2"))
        .withColumn("status", F.lit("Pending"))
        .drop("closest_permit_coordinates"))
 

    clustering.df_joined = (
        clustering.df_joined.withColumn(
            "detection_id", F.col("detection_id").cast("int")
        )
        .withColumn("object_lat", F.col("object_lat").cast("string"))
        .withColumn("object_lon", F.col("object_lon").cast("string"))
        .withColumn(
            "distance_closest_bridge", F.col("distance_closest_bridge").cast("float")
        )
        .withColumn("closest_bridge_id", F.col("closest_bridge_id").cast("string"))
        .withColumn(
            "distance_closest_permit", F.col("distance_closest_permit").cast("float")
        )
        .withColumn("closest_permit_lat", F.col("closest_permit_lat").cast("float"))
        .withColumn("closest_permit_lon", F.col("closest_permit_lon").cast("float"))
        .withColumn("score", F.col("score").cast("float"))
    )

    tableManager.write_to_table(
        clustering.df_joined, table_name="silver_objects_per_day"
    )
    tableManager.update_status(
        table_name="silver_frame_metadata", job_process_time=job_process_time
    )
    tableManager.update_status(
        table_name="silver_detection_metadata", job_process_time=job_process_time
    )

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
    config_file_path = os.path.join(project_root, "config.yml")
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
        job_process_time="2024-07-30 13:00:00",
    )
