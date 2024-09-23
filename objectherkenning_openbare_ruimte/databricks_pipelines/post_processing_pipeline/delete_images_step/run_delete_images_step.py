# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.frames import (  # noqa: E402
    BronzeFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.detections import (  # noqa: E402
    SilverDetectionMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.objects import (  # noqa: E402
    SilverObjectsPerDayManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    delete_file,
    setup_tables,
    unix_to_yyyy_mm_dd,
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
    job_date = job_process_time.split("T")[0]

    gps_internal_timestamp = unix_to_yyyy_mm_dd(
        BronzeFrameMetadataManager.get_gps_internal_timestamp_of_current_run(
            job_date=job_date
        )
    )
    stlanding_date_folder = unix_to_yyyy_mm_dd(gps_internal_timestamp)
    all_image_names = BronzeFrameMetadataManager.get_all_image_names_current_run(
        job_date=job_date
    )
    print(f"{len(all_image_names)} images found on {stlanding_date_folder}.")
    detection_ids = SilverObjectsPerDayManager.get_detection_ids_to_delete_current_run(
        job_date=job_date
    )
    to_keep_image_names = [
        SilverDetectionMetadataManager.get_image_name_from_detection_id(d)
        for d in detection_ids
    ]
    print(f"{len(to_keep_image_names)} images to keep.")

    to_delete_image_names = list(set(all_image_names) - set(to_keep_image_names))
    print(f"{len(to_delete_image_names)} images to delete.")

    successful_deletions = 0
    for img in to_delete_image_names:
        path = f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{stlanding_date_folder}/{img}"
        if delete_file(databricks_volume_full_path=path):
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
    run_delete_images_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        device_id=settings["device_id"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
    )
