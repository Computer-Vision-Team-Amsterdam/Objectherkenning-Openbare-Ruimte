# Run clustering
# enrich with Decos data and with Bridges data 
# prioritize based on score
# store results in tables.

# this fixes the caching issues, reimports all modules
dbutils.library.restartPython() 

from pyspark.sql import SparkSession
from helpers.clustering_detections import Clustering
from helpers.vulnerable_bridges_handler import VulnerableBridgesHandler
from helpers.decos_data_connector import DecosDataHandler

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
    ########## SETUP ##########    
    # Setup clustering
    clustering = Clustering(spark=sparkSession, date="D14M03Y2024")  
    containers_coordinates_geometry = clustering.get_containers_coordinates_geometry()
    
    # Setup bridges data
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"
    bridgesHandler = VulnerableBridgesHandler(file_path)
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()

    # Setup permit data
    az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
    db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
    db_name = "mdbdataservices"
 
    decosDataHandler = DecosDataHandler(az_tenant_id, db_host, db_name, db_port=5432)

    ######### ENRICHMENTS ###########

    # Enrich with bridges data
    closest_bridges_distances = VulnerableBridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
        bridges_locations_as_linestrings=bridgesHandler.get_bridges_coordinates_geometry(),
        containers_locations_as_points=clustering.get_containers_coordinates_geometry())
    
    clustering.add_column(column_name="closest_bridge_distance", values=closest_bridges_distances)

    # Enrich with decos data # TODO fix the date to correspond with clustering
    query = "SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"
    decosDataHandler.run(query)
    query_result_df = decosDataHandler.get_query_result_df()
    decosDataHandler.process_query_result()
    permit_distances, closest_permits = DecosDataHandler.calculate_distances_to_closest_permits(
        permits_locations_as_points=decosDataHandler.get_permits_coordinates_geometry(),
        permits_ids=decosDataHandler.get_permits_ids(),
        containers_locations_as_points=containers_coordinates_geometry)

    clustering.add_column(column_name="closest_permit_distance", values=permit_distances)
    clustering.add_column(column_name="closest_permit_id", values=closest_permits)

    # Enrich with score 

    scores = [
            float(calculate_score(closest_bridges_distances[idx], permit_distances[idx]))
            for idx in range(len(clustering.get_containers_coordinates()))
        ]
    clustering.add_column(column_name="score", values=scores)

    display(clustering.df_joined)



    

