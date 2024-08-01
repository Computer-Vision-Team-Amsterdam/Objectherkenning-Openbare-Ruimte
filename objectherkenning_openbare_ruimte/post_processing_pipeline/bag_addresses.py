dbutils.library.restartPython()  # noqa

from helpers.decos_data_connector import DecosDataHandler
from pyspark.sql import SparkSession  # noqa: E402

from datetime import datetime

spark = SparkSession.builder.appName("DataEnrichment").getOrCreate()


# Setup permit data
az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com" 
db_name = "mdbdataservices"

decosDataHandler = DecosDataHandler(spark, az_tenant_id, db_host, db_name, db_port=5432)

# Enrich with decos data
date_to_query = datetime.today().strftime('%Y-%m-%d')
query = f"SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"
print(f"Querying the database for date {date_to_query}...")
decosDataHandler.run(query)
decosDataHandler.process_query_result() 