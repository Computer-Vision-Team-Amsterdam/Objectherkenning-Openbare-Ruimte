import json
from datetime import datetime

from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (
    load_settings,
)


def get_databricks_environment(spark: SparkSession):
    """
    Returns Productie, Ontwikkel or None based on the tags set in the subscription
    """

    tags = spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags")
    try:
        tags_json = json.loads(tags)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

    environment_tag = next(
        (tag for tag in tags_json if tag.get("key") == "environment"), None
    )
    if environment_tag:
        environment = environment_tag.get("value")
        return environment
    else:
        raise ValueError("Databricks environment is not set.")


def get_catalog_name(spark: SparkSession):
    """
    Sets the catalog name based on the environment retrieved from Databricks cluster tags
    """

    environment = get_databricks_environment(spark)
    if environment == "Ontwikkel":
        catalog_name = "dpcv_dev"
    elif environment == "Productie":
        catalog_name = "dpcv_prd"

    return catalog_name


def get_job_process_time(is_first_pipeline_step):
    if is_first_pipeline_step:
        job_process_time_settings = load_settings("../../../config.yml")[
            "databricks_pipelines"
        ]["job_process_time"]
        if job_process_time_settings["auto"] == "true":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            print(
                f"Using custom job process time:{ job_process_time_settings['custom_job_process_time']}"
            )
            custom_job_process_time = datetime.strptime(
                job_process_time_settings["custom_job_process_time"],
                "%Y-%m-%d %H:%M:%S",
            )
            return custom_job_process_time
    else:
        if job_process_time_settings["auto"] == "true":
            raise ValueError(
                "Running pipeline step by step requires setting auto:false and custom_job_process_time to a valid YYYY-MM-DD HH:MM:SS"
            )
        else:
            custom_job_process_time = job_process_time_settings[
                "custom_job_process_time"
            ]
            job_process_time = dbutils.jobs.taskValues.get(  # type: ignore[name-defined] # noqa: F821
                taskKey="data-ingestion",
                key="job_process_time",
                default=custom_job_process_time,
                debugValue=custom_job_process_time,
            )
            return job_process_time
