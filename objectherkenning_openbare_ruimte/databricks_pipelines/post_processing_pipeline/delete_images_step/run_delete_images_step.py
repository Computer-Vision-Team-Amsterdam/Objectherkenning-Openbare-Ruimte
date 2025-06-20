# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from datetime import datetime  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    delete_file,
    get_databricks_environment,
    get_job_process_time,
    get_landingzone_folder_for_timestamp,
    setup_tables,
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
    spark_session: SparkSession,
    catalog: str,
    schema: str,
    device_id: str,
    job_process_time: datetime,
    min_score: float = 0.4,
) -> None:
    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)
    job_date = job_process_time.strftime("%Y-%m-%d")

    stlanding_image_folder = get_landingzone_folder_for_timestamp(
        BronzeFrameMetadataManager.get_gps_timestamp_at_date(job_date=job_date)
    )
    image_files_current_run = dbutils.fs.ls(f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{stlanding_image_folder}/")  # type: ignore[name-defined] # noqa: F821, F405
    print(f"{len(image_files_current_run)} images found on {stlanding_image_folder}.")

    # Must compare candidates for deletion and images to keep, since one image may have multiple detections.
    delete_candidate_image_names: set = {
        SilverDetectionMetadataManager.get_image_name_from_detection_id(d)
        for d in SilverEnrichedDetectionMetadataManager.get_detection_ids_candidates_for_deletion(
            job_date=job_date, score_threshold=min_score
        )
    }
    to_keep_image_names: set = {
        SilverDetectionMetadataManager.get_image_name_from_detection_id(d)
        for d in SilverEnrichedDetectionMetadataManager.get_detection_ids_to_keep_current_run(
            job_date=job_date, score_threshold=min_score
        )
    }
    to_delete_image_names = delete_candidate_image_names - to_keep_image_names
    print(f"{len(to_delete_image_names)} images to delete.")

    successful_deletions = 0
    for file in image_files_current_run:
        image_name = file.name

        if image_name in to_delete_image_names:
            print(f"Deleting {image_name}...")
            if delete_file(databricks_volume_full_path=file.path):
                successful_deletions += 1
    print(f"{successful_deletions} images successfully deleted.")


def main() -> None:
    spark_session = SparkSession.builder.appName("ImageDeletion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    databricks_environment = get_databricks_environment(spark_session)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_delete_images_step(
        spark_session=spark_session,
        catalog=settings["catalog"],
        schema=settings["schema"],
        device_id=settings["device_id"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
        min_score=settings["job_config"]["min_score"],
    )


if __name__ == "__main__":
    main()
