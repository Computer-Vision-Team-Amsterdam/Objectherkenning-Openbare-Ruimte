from datetime import datetime

from databricks.sdk.runtime import dbutils

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


def delete_file_or_folder(databricks_volume_full_path: str, recurse: bool = False):
    """
    Delete a file or folder and, optionally, all of its contents. If a file is
    specified, the recurse parameter is ignored. If a directory is specified, an
    error occurs when recurse is False and the directory is not empty.

    Parameters
    ----------
        databricks_volume_full_path (str)
            The full path of the file. The path should be within the /Volumes/ directory in Databricks.
        recurse (bool, default: False)
            Whether or not to delete all contents.

    Returns
    -------
    (bool): Whether or not the operation was successful.
    """
    result = False
    try:
        # if the file exists, remove it
        result = dbutils.fs.rm(databricks_volume_full_path, recurse=recurse)
    except Exception as e:
        if "DirectoryNotEmpty" in str(e):
            print(
                f"The folder {databricks_volume_full_path} is not empty while recurse is set to False."
            )
        else:
            # If there’s another type of error, raise it
            raise RuntimeError(f"An unexpected error occurred: {e}")
    return result


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


def get_landingzone_folder_for_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d")


def setup_tables(spark_session, catalog, schema):
    TableManager.spark_session = spark_session
    TableManager.catalog = catalog
    TableManager.schema = schema
