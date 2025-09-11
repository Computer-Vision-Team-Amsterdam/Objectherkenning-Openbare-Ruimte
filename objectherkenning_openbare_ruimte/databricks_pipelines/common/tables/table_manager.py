from abc import ABC
from datetime import datetime
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType


class TableManager(ABC):
    spark_session = None
    catalog = None
    schema = None
    table_name = None
    id_column = "id"

    @classmethod
    def update_status(
        cls,
        job_process_time: datetime,
        id_column: Optional[str] = None,
        exclude_ids: List[int] = [],
        only_ids: Optional[List[int]] = None,
    ):
        """
        Update `status` of "pending" rows to "processed". IDs included in
        `exclude_ids` will be ignored. If `only_ids` is specified, only those
        IDs will be updated.
        """
        if not id_column:
            id_column = cls.id_column

        table_full_name = cls.get_table_full_name()

        count_pending_query = f"""
        SELECT COUNT(*) as pending_count
        FROM {table_full_name}
        WHERE status = 'Pending'
        """  # nosec
        total_pending_before = TableManager.spark_session.sql(count_pending_query).collect()[0][  # type: ignore
            "pending_count"
        ]

        update_query = f"""
        UPDATE {table_full_name}
        SET status = 'Processed', processed_at = '{job_process_time}'
        WHERE status = 'Pending'
        """  # nosec
        if len(exclude_ids) > 0:
            exclude_ids_str = ", ".join(map(str, exclude_ids))
            update_query += f" AND {id_column} NOT IN ({exclude_ids_str})"
        if (only_ids is not None) and (len(only_ids) > 0):
            only_ids_str = ", ".join(map(str, only_ids))
            update_query += f" AND {id_column} IN ({only_ids_str})"

        # If only_ids is an empty list, it means NO ids should be updated.
        if (only_ids is not None) and (len(only_ids) == 0):
            total_pending_after = total_pending_before
        else:
            TableManager.spark_session.sql(update_query)  # type: ignore
            total_pending_after = TableManager.spark_session.sql(count_pending_query).collect()[0][  # type: ignore
                "pending_count"
            ]

        updated_rows = total_pending_before - total_pending_after
        print(
            f"Updated {updated_rows} 'Pending' rows to 'Processed' in {table_full_name}, {total_pending_after} rows remained 'Pending'."
        )

    @classmethod
    def get_table_full_name(cls) -> str:
        return f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"

    @classmethod
    def get_table(cls) -> DataFrame:
        """
        Loads the table from the catalog and schema.

        Returns:
        -------
        DataFrame
            A DataFrame containing the rows from the specified table.
        """
        full_table_name = cls.get_table_full_name()
        table_rows = TableManager.spark_session.table(full_table_name)  # type: ignore
        print(f"Loaded {table_rows.count()} rows from {full_table_name}.")
        return table_rows

    @classmethod
    def load_pending_rows_from_table(cls) -> DataFrame:
        """
        Loads all rows with a 'Pending' status from the table.

        Returns:
        -------
        DataFrame
            A DataFrame containing the rows with a 'Pending' status from the specified table.
        """
        table_rows = cls.get_table()
        pending_table_rows = table_rows.filter("status = 'Pending'")
        print(
            f"Filtered to {pending_table_rows.count()} 'Pending' rows from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}."
        )
        return pending_table_rows

    @classmethod
    def remove_fields_from_table_schema(cls, fields_to_remove: set) -> StructType:
        """
        This method loads the schema of the table, removes the fields
        listed in `fields_to_remove`, and returns the modified schema.

        Parameters:
        ----------
        fields_to_remove: A set of field names to be removed from the schema.

        Returns:
        -------
        StructType: The modified schema with the specified fields removed.
        """
        table_schema = cls.get_table().schema

        # Modify the schema by removing the specified fields
        modified_schema = StructType(
            [field for field in table_schema if field.name not in fields_to_remove]
        )

        return modified_schema

    @classmethod
    def insert_data(cls, df, mode="append"):
        df.write.mode(mode).saveAsTable(
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
        )
        print(f"Appended {df.count()} rows to {cls.table_name}.")
