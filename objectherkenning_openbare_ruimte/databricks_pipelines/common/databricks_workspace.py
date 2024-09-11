import json
from datetime import datetime

from databricks.sdk.runtime import *  # noqa: F403
from pyspark.sql import SparkSession


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


def get_job_process_time(job_process_time_settings, is_first_pipeline_step):
    current_timestamp = datetime.now()
    # use auto: True when triggering the pipeline as a workflow
    if job_process_time_settings["auto"] is True:
        if is_first_pipeline_step:
            print(
                f"Using automatic job process time: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
            )
            return current_timestamp
        else:
            job_process_time = dbutils.jobs.taskValues.get(  # type: ignore[name-defined] # noqa: F821, F405
                taskKey="data-ingestion",
                key="job_process_time",
                default=current_timestamp,
                debugValue=current_timestamp,
            )
            return job_process_time
    # use auto: False when triggering the pipeline step by step. Not recommended, can lead to unexpected errors. Option exists for debugging purposes.
    # Requires setting custom_job_process_time to a valid YYYY-MM-DD HH:MM:SS that has not been used before.
    else:
        print(
            f"Using custom job process time:{ job_process_time_settings['custom_job_process_time']}"
        )
        custom_job_process_time = job_process_time_settings["custom_job_process_time"]
        return custom_job_process_time