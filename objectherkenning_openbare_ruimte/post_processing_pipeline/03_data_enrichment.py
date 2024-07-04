# Run clustering
# enrich with Decos data and with Bridges data
# prioritize based on score
# store results in tables.

# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # noqa

from helpers.decos_data_connector import DecosDataHandler
from helpers.vulnerable_bridges_handler import VulnerableBridgesHandler
from helpers import utils_visualization
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, ArrayType
from shapely.geometry import Point
import ast
from shapely.wkt import loads as wkt_loads

from datetime import datetime
from helpers.clustering_detections import Clustering  # noqa: E402

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

def update_silver_status(catalog_name, table_name):
    # Update the status of the rows where status is 'Pending'
    update_query = f"""
    UPDATE {catalog_name}.oor.{table_name} SET status = 'Processed' WHERE status = 'Pending'
    """
    # Execute the update query
    spark.sql(update_query)
    print(f"02: Updated 'Pending' status to 'Processed' in {catalog_name}.oor.{table_name}.")


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataEnrichment").getOrCreate()
    ########## SETUP ##########
    # Setup clustering
    clustering = Clustering(spark=sparkSession)
    clustering.cluster_and_select_images()
    containers_coordinates_geometry = clustering.get_containers_coordinates_geometry()

    # Setup bridges data
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    vuln_bridges_rel_path = "vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"
    bridgesHandler = VulnerableBridgesHandler(spark, file_path)
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()

    # Setup permit data
    az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
    db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com" if clustering.catalog == "dpcv_dev" else "prd-bbn1-01-dbhost.postgres.database.azure.com"
    db_name = "mdbdataservices"

    decosDataHandler = DecosDataHandler(spark, az_tenant_id, db_host, db_name, db_port=5432)

    ######### ENRICHMENTS ###########
    print(f'Number of containers: {len(containers_coordinates_geometry)}. Number of vulnerable bridges: {len(bridgesHandler.get_bridges_coordinates())}.')
    # Enrich with bridges data
    closest_bridges_distances, closest_bridges_ids, closest_bridges_coordinates, closest_bridges_wkts = VulnerableBridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
        bridges_locations_as_linestrings=bridges_coordinates_geometry,
        containers_locations_as_points=containers_coordinates_geometry,
        bridges_ids=bridgesHandler.get_bridges_ids(),
        bridges_coordinates=bridgesHandler.get_bridges_coordinates()
    )

    clustering.add_column(column_name="closest_bridge_distance", values=closest_bridges_distances)
    clustering.add_column(column_name="closest_bridge_id", values=closest_bridges_ids)
    clustering.add_column(column_name="closest_bridge_coordinates", values=closest_bridges_coordinates)
    clustering.add_column(column_name="closest_bridge_linestring_wkt", values=closest_bridges_wkts)
    

    decosDataHandler = DecosDataHandler(spark, az_tenant_id, db_host, db_name, db_port=5432)

    # Enrich with decos data
    date_to_query = datetime.today().strftime('%Y-%m-%d')
    query = f"SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"
    print(f"Querying the database for date {date_to_query}...")
    decosDataHandler.run(query)
    decosDataHandler.process_query_result()
    

    permit_distances, closest_permits, closest_permits_coordinates = decosDataHandler.calculate_distances_to_closest_permits(
        permits_locations_as_points=decosDataHandler.get_permits_coordinates_geometry(),
        permits_ids=decosDataHandler.get_permits_ids(),
        permits_coordinates=decosDataHandler.get_permits_coordinates(),
        containers_locations_as_points=containers_coordinates_geometry
    )

    clustering.add_column(column_name="closest_permit_distance", values=permit_distances)
    clustering.add_column(column_name="closest_permit_id", values=closest_permits)
    clustering.add_column(column_name="closest_permit_coordinates", values=closest_permits_coordinates)
    
    # Enrich with score

    scores = [
            float(calculate_score(closest_bridges_distances[idx], permit_distances[idx]))
            for idx in range(len(clustering.get_containers_coordinates()))
        ]
    clustering.add_column(column_name="score", values=scores)


    # # - From here on, it's WIP -

    # Gather data to visualize
    vulnerable_bridges = wkt_loads(closest_bridges_wkts)
    permit_locations = [Point(x,y) for x,y in closest_permits_coordinates]
    detections = containers_coordinates_geometry
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f'{current_datetime}-map'
    path = f"/Volumes/dpcv_dev/default/landingzone/Luna/visualizations/{date_to_query}"

    utils_visualization.generate_map(
        dataframe=clustering.df_joined,
        name=name,
        path=path,
    )


    clustering.df_joined = clustering.df_joined.select(["detection_id", "object_class", "gps_lat", "gps_lon", "closest_bridge_distance", "closest_bridge_id", "closest_permit_distance", "closest_permit_id", "closest_permit_coordinates", "score"])

    clustering.df_joined = clustering.df_joined.withColumnRenamed("gps_lat", "object_lat").withColumnRenamed("gps_lon", "object_lon").withColumnRenamed("closest_bridge_distance", "distance_closest_bridge").withColumnRenamed("closest_permit_distance", "distance_closest_permit")

    clustering.df_joined = clustering.df_joined.withColumn("closest_permit_lat", F.col("closest_permit_coordinates._1"))
    clustering.df_joined = clustering.df_joined.withColumn("closest_permit_lon", F.col("closest_permit_coordinates._2"))
    clustering.df_joined =clustering.df_joined.withColumn("status", F.lit("Pending"))

    clustering.df_joined = clustering.df_joined.drop("closest_permit_coordinates")

    clustering.df_joined = (
        clustering.df_joined
        .withColumn("detection_id", F.col("detection_id").cast("int"))
        .withColumn("object_lat", F.col("object_lat").cast("string"))
        .withColumn("object_lon", F.col("object_lon").cast("string"))
        .withColumn("distance_closest_bridge", F.col("distance_closest_bridge").cast("float"))
        .withColumn("closest_bridge_id", F.col("closest_bridge_id").cast("string"))
        .withColumn("distance_closest_permit", F.col("distance_closest_permit").cast("float"))
        .withColumn("closest_permit_lat", F.col("closest_permit_lat").cast("float"))
        .withColumn("closest_permit_lon", F.col("closest_permit_lon").cast("float"))
        .withColumn("score", F.col("score").cast("float"))
    )

    # Store data in silver_object_per_day 
    clustering.df_joined.write.mode('append').saveAsTable(f'{clustering.catalog}.oor.silver_objects_per_day')
    print(f"03: Appended {clustering.df_joined.count()} rows to silver_objects_per_day.")

    update_silver_status(catalog_name=clustering.catalog, table_name="silver_frame_metadata")
    update_silver_status(catalog_name=clustering.catalog, table_name="silver_detection_metadata")