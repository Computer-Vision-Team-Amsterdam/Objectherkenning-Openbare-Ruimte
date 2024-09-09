from datetime import datetime


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
    except Exception as e:
        # Check if the error is due to the file not being found
        if "FileNotFoundException" in str(e):
            print(f"File {file_path} does not exist, so it cannot be removed.")
        else:
            # If there’s another type of error, raise it
            raise RuntimeError(f"An unexpected error occurred: {e}")


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

    # Fetch the image name based on the detection_id
    fetch_image_name_query = f"""
                            SELECT {catalog}.{schema}.silver_detection_metadata.image_name
                            FROM {catalog}.{schema}.silver_detection_metadata
                            WHERE {catalog}.{schema}.silver_detection_metadata.id = {detection_id}
                            """  # nosec
    image_name_result_df = spark.sql(fetch_image_name_query)  # noqa: F405

    # Extract the image name from the result
    image_basename = image_name_result_df.collect()[0]["image_name"]

    fetch_date_of_image_upload = f"""SELECT {catalog}.{schema}.silver_frame_metadata.gps_date FROM {catalog}.{schema}.silver_frame_metadata WHERE {catalog}.{schema}.silver_frame_metadata.image_name = '{image_basename}'"""  # nosec

    date_of_image_upload_df = spark.sql(fetch_date_of_image_upload)  # noqa: F405
    date_of_image_upload_dmy = date_of_image_upload_df.collect()[0]["gps_date"]

    # Convert to datetime object
    date_obj = datetime.strptime(date_of_image_upload_dmy, "%d/%m/%Y")

    # Format the datetime object to the desired format
    date_of_image_upload_ymd = date_obj.strftime("%Y-%m-%d")

    # Construct the path to the image to be uploaded to Signalen
    # TODO replace luna with 'device_id', not sure if to add it as class attribute
    image_upload_path = f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{date_of_image_upload_ymd}/{image_basename}"

    return image_upload_path
