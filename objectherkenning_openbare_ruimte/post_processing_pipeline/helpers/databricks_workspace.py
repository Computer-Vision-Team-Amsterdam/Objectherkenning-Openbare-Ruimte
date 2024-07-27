import json
from pyspark.sql import SparkSession
from datetime import datetime

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

    environment_tag = next((tag for tag in tags_json if tag.get("key") == "environment"), None)
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

def set_job_process_time():  
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
