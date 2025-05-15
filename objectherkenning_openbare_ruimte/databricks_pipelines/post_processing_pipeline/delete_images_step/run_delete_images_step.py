# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    delete_file,
    get_databricks_environment,
    get_job_process_time,
    setup_tables,
    unix_to_yyyy_mm_dd,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (  # noqa: E402
    BronzeFrameMetadataManager,
    SilverDetectionMetadataManager,
    SilverEnrichedDetectionMetadataManager,
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
    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)
    job_date = job_process_time.strftime("%Y-%m-%d")

    stlanding_image_folder = unix_to_yyyy_mm_dd(
        BronzeFrameMetadataManager.get_gps_timestamp_at_date(job_date=job_date)
    )
    image_files_current_run = dbutils.fs.ls(f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{stlanding_image_folder}/")  # type: ignore[name-defined] # noqa: F821, F405
    print(f"{len(image_files_current_run)} images found on {stlanding_image_folder}.")
    detection_ids = (
        SilverEnrichedDetectionMetadataManager.get_detection_ids_to_keep_current_run(
            job_date=job_date
        )
    )
    to_keep_image_names = [
        SilverDetectionMetadataManager.get_image_name_from_detection_id(d)
        for d in detection_ids
    ]
    print(f"{len(to_keep_image_names)} images to keep.")

    successful_deletions = 0
    for file in image_files_current_run:
        image_name = file.name

        if image_name not in to_keep_image_names:
            print(f"Deleting {image_name}...")
            if delete_file(databricks_volume_full_path=file.path):
                successful_deletions += 1
    print(f"{successful_deletions} images successfully deleted.")


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("ImageDeletion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_delete_images_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        device_id=settings["device_id"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
    )
