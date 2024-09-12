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

    def get_table(self, table_name: str) -> DataFrame:
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
        print(f"Loaded {table_rows.count()} rows from {full_table_name}.")
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
        table_rows = self.get_table(table_name)
        pending_table_rows = table_rows.filter("status = 'Pending'")
        print(
            f"Filtered to {pending_table_rows.count()} 'Pending' rows from {self.catalog}.{self.schema}.{table_name}."
        )
        return pending_table_rows

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
        table_schema = self.get_table(table_name=table_name).schema

        # Modify the schema by removing the specified fields
        modified_schema = StructType(
            [field for field in table_schema if field.name not in fields_to_remove]
        )

        return modified_schema

    def write_to_table(self, df, table_name, mode="append"):
        df.write.mode(mode).saveAsTable(f"{self.catalog}.{self.schema}.{table_name}")
        print(f"Appended {df.count()} rows to {table_name}.")
