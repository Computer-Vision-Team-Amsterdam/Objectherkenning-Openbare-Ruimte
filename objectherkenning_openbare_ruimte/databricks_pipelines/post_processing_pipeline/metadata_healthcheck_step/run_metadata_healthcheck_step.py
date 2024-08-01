# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.post_processing_pipeline.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
)
from objectherkenning_openbare_ruimte.post_processing_pipeline.helpers.metadata_healthcheck import (  # noqa: E402
    MetadataHealthChecker,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_metadata_healthcheck_step(sparkSession, catalog, schema, job_process_time):
    MetadataHealthChecker(sparkSession, catalog, schema, job_process_time)


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("MetadataHealthChecker").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings("../../config.yml")["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_metadata_healthcheck_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        job_process_time="2024-07-30 13:00:00",
    )
