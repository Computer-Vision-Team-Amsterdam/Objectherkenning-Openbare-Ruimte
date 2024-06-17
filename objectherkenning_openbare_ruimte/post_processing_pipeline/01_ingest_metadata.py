# this fixes the caching issues, reimports all modules
dbutils.library.restartPython() 

import tempfile
from pyspark.sql.functions import input_file_name, col, split, array, expr, regexp_extract
from pyspark.sql import SparkSession

from helpers.databricks_workspace import get_catalog_name

class DataLoader:
    
    def __init__(self, spark):
        self.spark = spark
        self.catalog = get_catalog_name(spark)
        self.schema = "oor"
   
        self.frame_metadata_table = f"{self.catalog}.{self.schema}.bronze_frame_metadata"
        self.detection_metadata_table = f"{self.catalog}.{self.schema}.bronze_detection_metadata"        
        self.root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net/Luna"
        self.checkpoint_path = f"{self.root_source}/_checkpoint"

        self._setup_initial_files()


    def _setup_initial_files(self):
        pass

    def _get_schema_path(self, table_name):
        """
        Retrieves the schema of the specified table and saves it to a temporary file.

        Parameters:
            table_name (str): The name of the table.

        Returns:
            str: The path to the temporary file containing the schema JSON.
        """
        # Retrieve the schema of the specified table
        existing_table_schema = self.spark.table(table_name).schema
        schema_json = existing_table_schema.json()

        # Save the JSON schema to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write(schema_json)
            path_table_schema = temp_file.name

        return path_table_schema

    def ingest_frame_metadata(self):
        
        source = f"{self.root_source}/frame_metadata"
        path_table_schema = self._get_schema_path(self.frame_metadata_table)

        df = self._load_new_frame_metadata(source, path_table_schema=path_table_schema, format="csv")
        self._store_new_data(df, target=self.frame_metadata_table)


    def ingest_detection_metadata(self):

        source = f"{self.root_source}/detection_metadata"
        path_table_schema = self._get_schema_path(self.detection_metadata_table)

        df = self._load_new_detection_metadata(source, path_table_schema, format="txt")
        self._store_new_data(df, target=self.detection_metadata_table)


    def _load_new_frame_metadata(self, source:str, path_table_schema:str, format:str):

        bronze_df = (self.spark.readStream \
                .format("cloudFiles") \
                .option("cloudFiles.format", format) \
                .option("cloudFiles.schemaLocation", path_table_schema) \
                .option("cloudFiles.inferColumnTypes", "true") \
                .option("cloudFiles.schemaHints", 'imu_pitch float, imu_roll float, imu_heading float, imu_gx float, imu_gy float, imu_gz float, gps_lat float, gps_lon float')  
                .option("cloudFiles.schemaEvolutionMode", "none")   
                .load(source)
                .withColumnRenamed("pylon://0_frame_counter", "pylon0_frame_counter")
                .withColumnRenamed("pylon://0_frame_timestamp", "pylon0_frame_timestamp"))
        return bronze_df
    
    def _load_new_detection_metadata(self, source:str):
        bronze_d1 = self.spark.read.option("recursiveFileLookup", "true") \
            .text(source) \
            .withColumn("filename", input_file_name())

        # Filter out rows not ending with .txt and extract filename
        bronze_d1 = bronze_d1.filter( bronze_d1["filename"].endswith(".txt")) \
                .withColumn("filename", regexp_extract(input_file_name(), r"/([^/]+)\.txt$", 1))


        # Split the 'value' column and create 'class' and 'bounding_box' columns
        bronze_d1 = bronze_d1.withColumn("split_col", split(bronze_d1["value"], " ")) \
                .withColumn("class", col("split_col")[0]) \
                .withColumn("bounding_box", array([col("split_col")[i].cast("float") for i in range(0, 4)])) \
                .drop("value", "split_col")


    def _store_new_data(self, df, target):
        stream_query = (df.writeStream 
            .option("checkpointLocation", self.checkpoint_path) 
            .trigger(availableNow=True) 
            .toTable(target))


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataIngestion").getOrCreate()
    dataLoader = DataLoader(sparkSession)
    df = dataLoader.ingest_frame_metadata()
