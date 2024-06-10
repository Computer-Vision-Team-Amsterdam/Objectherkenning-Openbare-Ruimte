def get_databricks_environment():
    """
    Returns Productie, Ontwikkel or None based on the tags set in the subscription
    """
    tags = spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags")
    tags_json = json.loads(tags)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

    environment_tag = next((tag for tag in tags_json if tag.get("key") == "environment"), None)

    if environment_tag:
        environment = environment_tag.get("value")
        return environment   
    return None 

def get_catalog_name():
    """
    Sets the catalog name based on the environment retrieved from Databricks cluster tags
    """

    environment = get_databricks_environment()
    if environment is None:
        raise ValueError("Databricks environment is not set.")
    if environment == "Ontwikkel":
        catalog_name = "dpcv_dev"
    elif environment == "Productie":
        catalog_name = "dpcv_prd"
   
    return catalog_name    