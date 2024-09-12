# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from datetime import datetime  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.functions import col, date_format  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.table_manager import (  # noqa: E402
    TableManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    delete_file,
    get_image_name_from_detection_id,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_delete_images_step(
    sparkSession,
    catalog,
    schema,
    device_id,
    job_process_time,
):
    job_date = job_process_time.split("T")[0]
    tableManager = TableManager(spark=sparkSession, catalog=catalog, schema=schema)

    bronze_frame_metadata_df = tableManager.load_from_table(
        table_name="bronze_frame_metadata"
    )
    filtered_df = bronze_frame_metadata_df.filter(
        (date_format(col("processed_at"), "yyyy-MM-dd") == job_date)
    )

    gps_date_value = filtered_df.select("gps_date").first()[0]
    all_image_names_current_run_list = (
        filtered_df.select("image_name").rdd.flatMap(lambda x: x).collect()
    )

    print(f"{len(all_image_names_current_run_list)} images found on {gps_date_value}.")

    silver_objects_per_day_df = tableManager.load_from_table(
        table_name="silver_objects_per_day"
    )

    # handle images from the current run only where score is Green (score > 1 as with the current definition)
    filtered_df = silver_objects_per_day_df.filter(
        (col("score") > 1)
        & (date_format(col("processed_at"), "yyyy-MM-dd") == job_date)
    )

    detection_ids = (
        filtered_df.select("detection_id").rdd.flatMap(lambda x: x).collect()
    )
    to_keep_image_names_current_run_list = []
    for detection_id in detection_ids:
        to_keep_image_name = get_image_name_from_detection_id(
            sparkSession, catalog, schema, detection_id
        )
        to_keep_image_names_current_run_list.append(to_keep_image_name)
    print(f"{len(to_keep_image_names_current_run_list)} images to keep.")

    # Substract image names we want to keep from all image names
    to_delete_image_names_current_run_list = list(
        set(all_image_names_current_run_list)
        - set(to_keep_image_names_current_run_list)
    )
    successful_deletions = 0
    formatted_gps_date_value = datetime.strptime(gps_date_value, "%d/%m/%Y").strftime(
        "%Y-%m-%d"
    )
    for img in to_delete_image_names_current_run_list:

        image_to_delete_full_path = f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{formatted_gps_date_value}/{img}"
        if delete_file(file_path=image_to_delete_full_path):
            successful_deletions += 1
    print(f"{successful_deletions} images successfully deleted.")


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("ImageDeletion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_db.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    job_process_time_settings = load_settings(config_file_path)["databricks_pipelines"][
        "job_process_time"
    ]
    run_delete_images_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        device_id=settings["device_id"],
        job_process_time=get_job_process_time(
            job_process_time_settings,
            is_first_pipeline_step=False,
        ),
    )
