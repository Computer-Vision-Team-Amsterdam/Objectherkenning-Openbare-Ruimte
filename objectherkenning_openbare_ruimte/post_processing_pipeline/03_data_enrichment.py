# Run clustering
# enrich with Decos data and with Bridges data
# prioritize based on score
# store results in tables.

# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

from datetime import datetime  # noqa: E402

from helpers import utils_visualization  # noqa: E402
from helpers.clustering_detections import Clustering  # noqa: E402
from helpers.databricks_workspace import get_databricks_environment  # noqa: E402
from helpers.decos_data_connector import DecosDataHandler  # noqa: E402
from helpers.table_manager import TableManager  # noqa: E402
from helpers.vulnerable_bridges_handler import VulnerableBridgesHandler  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_data_enrichment_step(
    sparkSession,
    catalog,
    schema,
    root_source,
    device_id,
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
        device_id=device_id,
        vuln_bridges_relative_path=vuln_bridges_relative_path,
    )
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()

    # Setup permit data
    decosDataHandler = DecosDataHandler(
        spark=SparkSession,
        az_tenant_id=az_tenant_id,
        db_host=db_host,
        db_name=db_name,
        db_port=5432,
    )

    tableManager = TableManager(spark=SparkSession, catalog=catalog, schema=schema)

    print(
        f"03: Number of containers: {len(containers_coordinates_geometry)}. Number of vulnerable bridges: {len(bridgesHandler.get_bridges_coordinates())}."
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

    clustering.add_column(
        column_name="closest_bridge_distance", values=closest_bridges_distances
    )
    clustering.add_column(column_name="closest_bridge_id", values=closest_bridges_ids)
    clustering.add_column(
        column_name="closest_bridge_coordinates", values=closest_bridges_coordinates
    )
    clustering.add_column(
        column_name="closest_bridge_linestring_wkt", values=closest_bridges_wkts
    )

    # Enrich with decos data
    date_to_query = datetime.today().strftime("%Y-%m-%d")
    query = f"SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"
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

    clustering.add_column(
        column_name="closest_permit_distance", values=permit_distances
    )
    clustering.add_column(column_name="closest_permit_id", values=closest_permits)
    clustering.add_column(
        column_name="closest_permit_coordinates", values=closest_permits_coordinates
    )

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

    clustering.df_joined = clustering.df_joined.select(
        [
            "detection_id",
            "object_class",
            "gps_lat",
            "gps_lon",
            "closest_bridge_distance",
            "closest_bridge_id",
            "closest_permit_distance",
            "closest_permit_id",
            "closest_permit_coordinates",
            "score",
        ]
    )

    clustering.df_joined = (
        clustering.df_joined.withColumnRenamed("gps_lat", "object_lat")
        .withColumnRenamed("gps_lon", "object_lon")
        .withColumnRenamed("closest_bridge_distance", "distance_closest_bridge")
        .withColumnRenamed("closest_permit_distance", "distance_closest_permit")
    )

    clustering.df_joined = clustering.df_joined.withColumn(
        "closest_permit_lat", F.col("closest_permit_coordinates._1")
    )
    clustering.df_joined = clustering.df_joined.withColumn(
        "closest_permit_lon", F.col("closest_permit_coordinates._2")
    )
    clustering.df_joined = clustering.df_joined.withColumn("status", F.lit("Pending"))

    clustering.df_joined = clustering.df_joined.drop("closest_permit_coordinates")

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

    # Store data in silver_object_per_day
    clustering.df_joined.write.mode("append").saveAsTable(
        f"{clustering.catalog}.oor.silver_objects_per_day"
    )
    print(
        f"03: Appended {clustering.df_joined.count()} rows to silver_objects_per_day."
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
    settings = load_settings("../../config.yml")["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_data_enrichment_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        root_source=settings["storage_account_root_path"],
        device_id=settings["device_id"],
        vuln_bridges_relative_path=settings["vuln_bridges_relative_path"],
        az_tenant_id=settings["azure_tenant_id"],
        db_host=settings["reference_database"]["host"],
        db_name=settings["reference_database"]["name"],
        job_process_time="2024-07-30 13:00:00",
    )
