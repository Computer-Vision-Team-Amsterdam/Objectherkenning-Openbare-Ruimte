# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

import requests  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.decos_data_connector import (  # noqa: E402
    DecosDataHandler,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.ingest_metadata_step.components.data_ingestion import (  # noqa: E402
    DataLoader,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_ingest_metadata_step(
    sparkSesssion,
    catalog,
    schema,
    root_source,
    device_id,
    ckpt_frames_relative_path,
    ckpt_detections_relative_path,
    job_process_time,
    settings,
):
    def healthcheck(sparkSession, settings):
        decosDataHandler = DecosDataHandler(
            spark=sparkSession,
            az_tenant_id=settings["azure_tenant_id"],
            db_host=settings["reference_database"]["host"],
            db_name=settings["reference_database"]["name"],
            db_port=5432,
            active_object_classes=settings["object_classes"]["active"],
            permit_mapping=settings["object_classes"]["permit_mapping"],
        )
        result = decosDataHandler.get_benkagg_adresseerbareobjecten_by_id(
            "0363200000006110"
        )
        print(result[0])
        print(result[0].columns)
        bag_url = (
            f"https://api.data.amsterdam.nl/geosearch/?datasets=benkagg/adresseerbareobjecten"
            f"&lat={'52.3782197'}&lon={'4.8834705'}&radius=25"
        )

        response = requests.get(bag_url, timeout=60)
        if response.status_code == 200:
            print("BAG API is up and running")
        else:
            print("BAG API is down")

    healthcheck(sparkSesssion, settings)

    dataLoader = DataLoader(
        sparkSesssion,
        catalog,
        schema,
        root_source,
        device_id,
        ckpt_frames_relative_path,
        ckpt_detections_relative_path,
        job_process_time,
    )
    dataLoader.ingest_frame_metadata()
    dataLoader.ingest_detection_metadata()
    dbutils.jobs.taskValues.set(  # type: ignore[name-defined] # noqa: F821
        key="job_process_time", value=dataLoader.job_process_time.isoformat()
    )

    # Cleanup temporary files
    dataLoader.cleanup_temp_files()


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataIngestion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    run_ingest_metadata_step(
        sparkSesssion=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        root_source=settings["storage_account_root_path"],
        device_id=settings["device_id"],
        ckpt_frames_relative_path=settings["ckpt_frames_relative_path"],
        ckpt_detections_relative_path=settings["ckpt_detections_relative_path"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=True,
        ),
        settings=settings,
    )
