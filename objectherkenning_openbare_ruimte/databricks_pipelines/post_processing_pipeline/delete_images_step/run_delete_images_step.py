# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

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
    get_image_upload_path_from_detection_id,
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
    job_date = job_process_time.date()
    tableManager = TableManager(spark=sparkSession, catalog=catalog, schema=schema)
    silver_objects_per_day_df = tableManager.load_from_table(
        table_name="silver_objects_per_day"
    )

    # handle images from the current run only where score is Green (score > 1 as with the current definition)
    filtered_df = silver_objects_per_day_df.filter(
        (col("score") > 1)
        & (date_format(col("processed_at"), "yyyy-MM-dd") == job_date)
    )
    for row in filtered_df.collect():
        # images could have been uploaded on a different date, the function below accounts for this
        image_to_delete_full_path = get_image_upload_path_from_detection_id(
            spark=sparkSession,
            catalog=catalog,
            schema=schema,
            detection_id=row["detection_id"],
            device_id=device_id,
        )
        delete_file(file_path=image_to_delete_full_path)


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("ImageDeletion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config.yml")
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
