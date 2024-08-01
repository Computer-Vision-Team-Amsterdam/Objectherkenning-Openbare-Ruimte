# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.metadata_healthcheck_step.components.metadata_healthcheck import (  # noqa: E402
    MetadataHealthChecker,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_metadata_healthcheck_step(sparkSession, catalog, schema, job_process_time):
    MetadataHealthChecker(sparkSession, catalog, schema, job_process_time)


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("MetadataHealthChecker").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_metadata_healthcheck_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        job_process_time="2024-07-30 13:00:00",
    )