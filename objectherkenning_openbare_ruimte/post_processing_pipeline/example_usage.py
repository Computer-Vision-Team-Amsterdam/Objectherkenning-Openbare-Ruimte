# this fixes the caching issues
dbutils.library.restartPython() 

from pyspark.sql import SparkSession
from helpers.decos_data_connector import DecosDataHandler
from helpers.vulnerable_bridges_handler import VulnerableBridgesHandler


def decos():
    # Setup permit data
    az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
    db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
    db_name = "mdbdataservices"
 
    decosDataHandler = DecosDataHandler(spark, az_tenant_id, db_host, db_name, db_port=5432)
    query = "SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"
    decosDataHandler.run(query)
    query_result_df = decosDataHandler.get_query_result_df()
    display(query_result_df)
    decosDataHandler.process_query_result()
    healthy_df = decosDataHandler.get_healthy_df()
    display(healthy_df)
    coords_geom = decosDataHandler.get_permits_coordinates_geometry()
    print(coords_geom)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Example").getOrCreate()

    # Setup bridges data
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"
    bridgesHandler = VulnerableBridgesHandler(spark, file_path)
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()
    print(bridges_coordinates_geometry)
