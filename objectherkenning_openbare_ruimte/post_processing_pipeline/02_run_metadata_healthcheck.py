# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()

from pyspark.sql.functions import col
from helpers.databricks_workspace import get_catalog_name
from pyspark.sql import SparkSession

# Read the job process time from the first task's output
# job_process_time = dbutils.jobs.taskValues.get(taskKey = "data-ingestion", key = "job_process_time", default = 0, debugValue=0)

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
        job_process_time="2024-07-30 13:00:00"
    )
    