from datetime import datetime

from databricks.sdk.runtime import *  # noqa: F403


def delete_file(file_path):
    """
    Delete file from storage account.

    Parameters:
    file_path (str): The full path of the file using the volumes. The path should be within the /Volumes/ directory in Databricks.

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
            print(f"File {file_path} does not exist, so it cannot be removed.")
            return False
        else:
            # If there’s another type of error, raise it
            raise RuntimeError(f"An unexpected error occurred: {e}")


def get_image_name_from_detection_id(spark, catalog, schema, detection_id):

    fetch_image_name_query = f"""
                            SELECT {catalog}.{schema}.silver_detection_metadata.image_name
                            FROM {catalog}.{schema}.silver_detection_metadata
                            WHERE {catalog}.{schema}.silver_detection_metadata.id = {detection_id}
                            """  # nosec
    image_name_result_df = spark.sql(fetch_image_name_query)  # noqa: F405

    image_basename = image_name_result_df.collect()[0]["image_name"]
    return image_basename


def get_image_upload_path_from_detection_id(
    spark, catalog, schema, detection_id, device_id
):
    """
    Fetches the image name based on the detection_id and constructs the path for uploading the image.

    Parameters:
    detection_id (int): The id of the detection.
    device_id (str): Used to construct the full path

    Returns:
    str: The constructed image upload path.
    """

    image_basename = get_image_name_from_detection_id(
        spark=spark,
        catalog=catalog,
        schema=schema,
        detection_id=detection_id,
    )

    fetch_date_of_image_upload = f"""SELECT {catalog}.{schema}.silver_frame_metadata.gps_date FROM {catalog}.{schema}.silver_frame_metadata WHERE {catalog}.{schema}.silver_frame_metadata.image_name = '{image_basename}'"""  # nosec

    date_of_image_upload_df = spark.sql(fetch_date_of_image_upload)  # noqa: F405
    date_of_image_upload_dmy = date_of_image_upload_df.collect()[0]["gps_date"]

    # Convert to datetime object
    date_obj = datetime.strptime(date_of_image_upload_dmy, "%d/%m/%Y")

    # Format the datetime object to the desired format
    date_of_image_upload_ymd = date_obj.strftime("%Y-%m-%d")

    # Construct the path to the image to be uploaded to Signalen
    image_upload_path = f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{date_of_image_upload_ymd}/{image_basename}"

    return image_upload_path


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
