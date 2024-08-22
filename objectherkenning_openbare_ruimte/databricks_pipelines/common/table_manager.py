from datetime import datetime

from pyspark.sql import SparkSession


class TableManager:
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema

    def update_status(
        self, table_name: str, job_process_time: datetime, exclude_ids=[]
    ):
        count_pending_query = f"""
        SELECT COUNT(*) as pending_count
        FROM {self.catalog}.{self.schema}.{table_name}
        WHERE status = 'Pending'
        """
        total_pending_before = self.spark.sql(count_pending_query).collect()[0][
            "pending_count"
        ]

        update_query = f"""
        UPDATE {self.catalog}.{self.schema}.{table_name}
        SET status = 'Processed', processed_at = '{job_process_time}'
        WHERE status = 'Pending'
        """
        if exclude_ids:
            exclude_ids_str = ", ".join(map(str, exclude_ids))
            update_query += f" AND id NOT IN ({exclude_ids_str})"

        self.spark.sql(update_query)

        total_pending_after = self.spark.sql(count_pending_query).collect()[0][
            "pending_count"
        ]
        updated_rows = total_pending_before - total_pending_after

        print(
            f"Updated {updated_rows} 'Pending' rows to 'Processed' in {self.catalog}.{self.schema}.{table_name}, {total_pending_after} rows remained 'Pending'."
        )

    def write_to_table(self, df, table_name, mode="append"):
        df.write.mode(mode).saveAsTable(f"{self.catalog}.{self.schema}.{table_name}")
        print(f"02: Appended {df.count()} rows to {table_name}.")    

    @staticmethod
    def compare_dataframes(df1, df2, df1_name, df2_name):
        print(f"Comparing dataframes {df1_name} and {df2_name}.")
        print(50*"-")    

        same_count = df1.count() == df2.count()
        print(f"Same number of rows: {same_count}")

        diff1 = df1.subtract(df2)
        diff2 = df2.subtract(df1)

        if diff1.count() == 0 and diff2.count() == 0:
            print("The DataFrames have the same content.")
        else:
            print("The DataFrames differ.")

        same_schema = df1.schema == df2.schema
        if same_schema:
            print("Same schema.")
        else:
            print("\nSchemas differ. Here are the details:")
            print(f"Schema of {df1_name}:")
            df1.printSchema()
            print(f"Schema of {df2_name}:")
            df2.printSchema()
        print(50*"-")       
