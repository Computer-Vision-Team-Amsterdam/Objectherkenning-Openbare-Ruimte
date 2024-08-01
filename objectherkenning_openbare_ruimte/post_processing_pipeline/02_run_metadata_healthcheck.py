# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()
from datetime import datetime

from pyspark.sql.functions import col
from helpers.databricks_workspace import get_catalog_name, get_databricks_environment
from helpers.metadata_healthcheck import MetadataHealthChecker
from pyspark.sql import SparkSession
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import load_settings

# Read the job process time from the first task's output
job_process_time = dbutils.jobs.taskValues.get(taskKey = "data-ingestion", key = "job_process_time", default = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), debugValue=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def run_metadata_healthcheck_step(sparkSession, catalog, schema, job_process_time):
    metadataHealthChecker = MetadataHealthChecker(sparkSession, catalog, schema, job_process_time)

if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("MetadataHealthChecker").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings("../../config.yml")["databricks_pipelines"][f"{databricks_environment}"]
    run_metadata_healthcheck_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        job_process_time=job_process_time
    )
    