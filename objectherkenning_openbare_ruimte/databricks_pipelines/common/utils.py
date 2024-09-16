from databricks.sdk.runtime import *  # noqa: F403


def delete_file(databricks_volume_full_path):
    """
    Delete file from storage account.

    Parameters:
    databricks_volume_full_path (str): The full path of the file using the volumes. The path should be within the /Volumes/ directory in Databricks.

    """
    try:
        # list the file to check if it exists
        dbutils.fs.ls(file_path)  # type: ignore[name-defined] # noqa: F821, F405
        # if the file exists, remove it
        dbutils.fs.rm(file_path)  # type: ignore[name-defined] # noqa: F821, F405
        return True
    except Exception as e:
        # Check if the error is due to the file not being found
        if "FileNotFoundException" in str(e):
            print(
                f"File {databricks_volume_full_path} does not exist, so it cannot be removed."
            )
            return False
        else:
            # If there’s another type of error, raise it
            raise RuntimeError(f"An unexpected error occurred: {e}")


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
