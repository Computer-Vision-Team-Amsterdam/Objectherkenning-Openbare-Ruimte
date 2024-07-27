from pyspark.sql import SparkSession
from datetime import datetime

class TableManager:
    def __init__(self, spark: SparkSession, catalog: str):
        self.spark = spark
        self.catalog = catalog

    def update_status(self, table_name: str, job_process_time: datetime):
        update_query = f"""
        UPDATE {self.catalog}.oor.{table_name} 
        SET status = 'Processed', processed_at = '{job_process_time}' 
        WHERE status = 'Pending'
        """
        self.spark.sql(update_query)
        print(f"Updated 'Pending' status to 'Processed' in {self.catalog}.oor.{table_name}.")