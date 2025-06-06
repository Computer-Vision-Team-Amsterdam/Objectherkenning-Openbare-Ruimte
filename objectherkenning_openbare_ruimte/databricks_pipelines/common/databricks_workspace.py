import json
from datetime import datetime
from typing import Union

from databricks.sdk.runtime import *  # noqa: F403
from pyspark.sql import SparkSession


def get_databricks_environment(spark_session: SparkSession) -> Union[str, None]:
    """
    Returns Productie, Ontwikkel or None based on the tags set in the subscription
    """

    tags = spark_session.conf.get("spark.databricks.clusterUsageTags.clusterAllTags")
    try:
        tags_json = json.loads(tags)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

    for tag in tags_json:
        if tag.get("key") == "environment":
            environment = tag.get("value")
            return environment
    raise ValueError("Databricks environment is not set.")


def get_job_process_time(is_first_pipeline_step: bool) -> datetime:
    current_timestamp = datetime.now()

    if is_first_pipeline_step:
        print(f"Job process time: {type(current_timestamp)} {current_timestamp}")
        return current_timestamp

    job_process_time = dbutils.jobs.taskValues.get(  # type: ignore[name-defined] # noqa: F821, F405
        taskKey="data-ingestion",
        key="job_process_time",
        default=current_timestamp,
        debugValue=current_timestamp,
    )
    if isinstance(job_process_time, str):
        if "T" in job_process_time:
            job_process_time = datetime.strptime(
                job_process_time, "%Y-%m-%dT%H:%M:%S.%f"
            )
        else:
            job_process_time = datetime.strptime(
                job_process_time, "%Y-%m-%d %H:%M:%S.%f"
            )

    print(f"Job process time: {type(job_process_time)} {job_process_time}")
    return job_process_time
