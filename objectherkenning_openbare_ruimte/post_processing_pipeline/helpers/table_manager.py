from pyspark.sql import SparkSession
from datetime import datetime

class TableManager:
    def __init__(self, spark: SparkSession, catalog: str):
        self.spark = spark
        self.catalog = catalog

    def update_status(self, table_name: str, job_process_time: datetime, exclude_ids = []):
        count_pending_query = f"""
        SELECT COUNT(*) as pending_count
        FROM {self.catalog}.oor.{table_name}
        WHERE status = 'Pending'
        """
        total_pending_before = self.spark.sql(count_pending_query).collect()[0]['pending_count']
        
        if exclude_ids:
            exclude_ids_str = ', '.join(map(str, exclude_ids))
            count_excluded_query = f"""
            SELECT COUNT(*) as excluded_count
            FROM {self.catalog}.oor.{table_name}
            WHERE status = 'Pending' AND id IN ({exclude_ids_str})
            """
            excluded_count = self.spark.sql(count_excluded_query).collect()[0]['excluded_count']
        else:
            excluded_count = 0
        
        update_query = f"""
        UPDATE {self.catalog}.oor.{table_name} 
        SET status = 'Processed', processed_at = '{job_process_time}' 
        WHERE status = 'Pending'
        """
        if exclude_ids:
            update_query += f" AND id NOT IN ({exclude_ids_str})"

        self.spark.sql(update_query)
        
        total_pending_after = self.spark.sql(count_pending_query).collect()[0]['pending_count']
        updated_rows = total_pending_before - total_pending_after
        
        print(f"Updated {updated_rows} 'Pending' rows to 'Processed' in {self.catalog}.oor.{table_name}, {total_pending_after} rows remained 'Pending'.")