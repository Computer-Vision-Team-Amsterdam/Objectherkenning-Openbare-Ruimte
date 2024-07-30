# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from pyspark.sql.functions import col
from helpers.databricks_workspace import get_catalog_name
from pyspark.sql import SparkSession

# Read the job process time from the first task's output
# job_process_time = dbutils.jobs.taskValues.get(taskKey = "data-ingestion", key = "job_process_time", default = 0, debugValue=0)
job_process_time = "2024-07-30 13:00:00"
print(f"Job process time: {job_process_time}")

class MetadataHealthChecker:
    def __init__(self, spark):
        self.spark = spark
        self.catalog = get_catalog_name(spark)
        # load bronze metadata that is pending
        self.bronze_frame_metadata = self.load_bronze_metadata(table_name="bronze_frame_metadata")
        print(f"02: Loaded {self.bronze_frame_metadata.count()} 'Pending' rows from {self.catalog}.oor.bronze_frame_metadata.")
        self.process_and_save_frame_metadata()
        self.update_bronze_status(table_name="bronze_frame_metadata")


        self.bronze_detection_metadata = self.load_bronze_metadata(table_name="bronze_detection_metadata")
        print(f"02: Loaded {self.bronze_detection_metadata.count()} 'Pending' rows from {self.catalog}.oor.bronze_detection_metadata.")

        self.process_and_save_detection_metadata()
        self.update_bronze_status(table_name="bronze_detection_metadata")


    def load_bronze_metadata(self, table_name):
        query = f"SELECT * FROM {self.catalog}.oor.{table_name} WHERE status='Pending'"
        return self.spark.sql(query)


    def process_and_save_frame_metadata(self):

        # Minimal check, we should implement more logic here
        valid_metadata = self.bronze_frame_metadata.filter((col('gps_lat').isNotNull()) & (col('gps_lat') != 0) &
                                    (col('gps_lon').isNotNull()) & (col('gps_lon') != 0))
        
        invalid_metadata = self.bronze_frame_metadata.filter((col('gps_lat').isNull()) | (col('gps_lat') == 0) |
                                        (col('gps_lon').isNull()) | (col('gps_lon') == 0))
        
        print("02: Processed frame metadata.")

        valid_metadata.write.mode('append').saveAsTable(f'{self.catalog}.oor.silver_frame_metadata')
        print(f"02: Appended {valid_metadata.count()} rows to {self.catalog}.oor.silver_frame_metadata.")

        invalid_metadata.write.mode('append').saveAsTable(f'{self.catalog}.oor.silver_frame_metadata_quarantine')
        print(f"02: Appended {invalid_metadata.count()} rows to {self.catalog}.oor.silver_frame_metadata_quarantine.")

    def process_and_save_detection_metadata(self):

        # Detection metadata corresponding to healthy frame metadata is healthy
        valid_metadata_query = f"""
                        SELECT {self.catalog}.oor.bronze_detection_metadata.*
                        FROM {self.catalog}.oor.bronze_detection_metadata
                        INNER JOIN {self.catalog}.oor.silver_frame_metadata ON {self.catalog}.oor.bronze_detection_metadata.image_name = {self.catalog}.oor.silver_frame_metadata.image_name
                        WHERE {self.catalog}.oor.bronze_detection_metadata.status = 'Pending'
                        """
        valid_metadata = self.spark.sql(valid_metadata_query)

        # Detection metadata corresponding to unhealthy frame metadata is unhealthy
        invalid_metadata_query = f"""
                        SELECT {self.catalog}.oor.bronze_detection_metadata.*
                        FROM {self.catalog}.oor.bronze_detection_metadata
                        INNER JOIN {self.catalog}.oor.silver_frame_metadata_quarantine ON {self.catalog}.oor.bronze_detection_metadata.image_name = {self.catalog}.oor.silver_frame_metadata_quarantine.image_name
                        WHERE {self.catalog}.oor.bronze_detection_metadata.status = 'Pending'
                        """

        invalid_metadata = self.spark.sql(invalid_metadata_query)
        print("02: Processed detection metadata.")

        valid_metadata.write.mode('append').saveAsTable(f'{self.catalog}.oor.silver_detection_metadata')
        print(f"02: Appended {valid_metadata.count()} rows to silver_detection_metadata.")

        invalid_metadata.write.mode('append').saveAsTable(f'{self.catalog}.oor.silver_detection_metadata_quarantine')
        print(f"02: Appended {invalid_metadata.count()} rows to silver_detection_metadata_quarantine.")

        
    def update_bronze_status(self, table_name):
        # Update the status of the rows where status is 'Pending'
        update_query = f"""
        UPDATE {self.catalog}.oor.{table_name} SET status = 'Processed',
        processed_at = '{job_process_time}' WHERE status = 'Pending'
        """
        # Execute the update query
        self.spark.sql(update_query)
        print(f"02: Updated 'Pending' status to 'Processed' in {self.catalog}.oor.{table_name}.")

if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("MetadataHealthChecker").getOrCreate()
    metadataHealthChecker = MetadataHealthChecker(sparkSession)
    