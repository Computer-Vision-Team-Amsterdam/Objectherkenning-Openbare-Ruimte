# this fixes the caching issues
dbutils.library.restartPython() 

from pyspark.sql import SparkSession
from helpers.decos_data_connector import DecosDataHandler


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Example").getOrCreate()

    # Setup permit data
    az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
    db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
    db_name = "mdbdataservices"
 
    decosDataHandler = DecosDataHandler(spark, az_tenant_id, db_host, db_name, db_port=5432)
    query = "SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"
    decosDataHandler.run(query)
    query_result_df = decosDataHandler.get_query_result_df()
    #decosDataHandler.display_dataframe(query_result_df)
    decosDataHandler.process_query_result()
    healthy_df = decosDataHandler.get_healthy_df()
    display(healthy_df)