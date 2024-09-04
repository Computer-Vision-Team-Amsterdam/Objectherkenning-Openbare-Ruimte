from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


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
        """  # nosec
        total_pending_before = self.spark.sql(count_pending_query).collect()[0][
            "pending_count"
        ]

        update_query = f"""
        UPDATE {self.catalog}.{self.schema}.{table_name}
        SET status = 'Processed', processed_at = '{job_process_time}'
        WHERE status = 'Pending'
        """  # nosec
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

    def _load_table(self, table_name: str) -> DataFrame:
        """
        Loads a table from the catalog and schema.

        Parameters:
        ----------
        table_name : str
            The name of the table to load.

        Returns:
        -------
        DataFrame
            A DataFrame containing the rows from the specified table.
        """
        full_table_name = f"{self.catalog}.{self.schema}.{table_name}"
        table_rows = self.spark.table(full_table_name)
        print(f"01: Loaded {table_rows.count()} rows from {full_table_name}.")
        return table_rows

    def load_pending_rows_from_table(self, table_name: str) -> DataFrame:
        """
        Loads all rows with a 'Pending' status from the specified table in the catalog and schema.

        Parameters:
        ----------
        table_name : str
            The name of the table from which to load 'Pending' rows.

        Returns:
        -------
        DataFrame
            A DataFrame containing the rows with a 'Pending' status from the specified table.
        """
        table_rows = self._load_table(table_name)
        pending_table_rows = table_rows.filter("status = 'Pending'")
        print(
            f"01: Filtered to {pending_table_rows.count()} 'Pending' rows from {self.catalog}.{self.schema}.{table_name}."
        )
        return pending_table_rows

    def load_from_table(self, table_name: str) -> DataFrame:
        """
        Loads all rows from the specified table in the catalog and schema.

        Parameters:
        ----------
        table_name: The name of the table to load.

        Returns:
        -------
        DataFrame: A DataFrame containing all rows from the specified table.
        """
        return self._load_table(table_name)

    def remove_fields_from_table_schema(
        self, table_name: str, fields_to_remove: set
    ) -> StructType:
        """
        This method loads the schema of the specified table, removes the fields
        listed in `fields_to_remove`, and returns the modified schema.

        Parameters:
        ----------
        table_name: The name of the table whose schema will be modified.
        fields_to_remove: A set of field names to be removed from the schema.

        Returns:
        -------
        StructType: The modified schema with the specified fields removed.

        """
        table_schema = self._load_table(table_name=table_name).schema

        # Modify the schema by removing the specified fields
        modified_schema = StructType(
            [field for field in table_schema if field.name not in fields_to_remove]
        )

        return modified_schema

    # def update_status(
    #     self, table_name: str, job_process_time: datetime, exclude_ids=[]
    # ):
    #     table_df = self.spark.table(f"{self.catalog}.{self.schema}.{table_name}")

    #     pending_df = table_df.filter(
    #         (col("status") == "Pending") & (~col("id").isin(exclude_ids))
    #     )

    #     total_pending_before = pending_df.count()

    #     updated_df = pending_df.withColumn("status", lit("Processed")).withColumn(
    #         "processed_at", lit(job_process_time)
    #     )

    #     updated_df.write.mode("overwrite").insertInto(
    #         f"{self.catalog}.{self.schema}.{table_name}"
    #     )

    #     table_df = self.spark.table(f"{self.catalog}.{self.schema}.{table_name}")
    #     total_pending_after = table_df.filter(col("status") == "Pending").count()

    #     updated_rows = total_pending_before - total_pending_after

    #     print(
    #         f"Updated {updated_rows} 'Pending' rows to 'Processed' in {self.catalog}.{self.schema}.{table_name}, {total_pending_after} rows remained 'Pending'."
    #     )

    def write_to_table(self, df, table_name, mode="append"):
        df.write.mode(mode).saveAsTable(f"{self.catalog}.{self.schema}.{table_name}")
        print(f"Appended {df.count()} rows to {table_name}.")

    @staticmethod
    def compare_dataframes(df1, df2, df1_name, df2_name):
        print(f"Comparing dataframes {df1_name} and {df2_name}.")
        print(50 * "-")

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
        print(50 * "-")
