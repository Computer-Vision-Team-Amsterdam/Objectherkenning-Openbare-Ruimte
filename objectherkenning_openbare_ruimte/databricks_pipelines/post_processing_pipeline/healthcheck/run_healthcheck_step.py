import os

import requests
from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    SignalConnectionConfigurer,
    get_databricks_environment,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step import (
    BENKAGGConnector,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (
    load_settings,
)


def run_healthcheck_refdb(settings: dict) -> None:
    benkAggConnector = BENKAGGConnector(
        az_tenant_id=settings["azure_tenant_id"],
        db_host=settings["reference_database"]["host"],
        db_name=settings["reference_database"]["name"],
    )
    result = benkAggConnector.get_benkagg_adresseerbareobjecten_by_id(
        "0363200000006110"
    )
    if result.iloc[0]["postcode"] == "1015NR":
        print("RefDB benk_agg data handler is up and running")
    else:
        raise ValueError("RefDB benk_agg data handler is down")


def run_healthcheck_api() -> None:
    bag_url = (
        "https://api.data.amsterdam.nl/geosearch/?datasets=benkagg/adresseerbareobjecten"
        "&lat=52.3782197&lon=4.8834705&radius=25"
    )
    try:
        response = requests.get(bag_url, timeout=60)
        response.raise_for_status()
        print("BAG API is up and running")
    except requests.RequestException as e:
        raise ValueError("BAG API is down") from e


def run_healthcheck_sia(settings: dict) -> None:
    signalConnectionConfigurer = SignalConnectionConfigurer(
        client_id=settings["signalen"]["client_id"],
        client_secret_name=settings["signalen"]["client_secret_name"],
        access_token_url=settings["signalen"]["access_token_url"],
        base_url=settings["signalen"]["base_url"],
    )
    try:
        _ = signalConnectionConfigurer.get_access_token()
        print("SIA token successful")
    except requests.RequestException as e:
        raise ValueError("SIA token unsuccessful") from e


def main():
    spark_session = SparkSession.builder.appName("DataIngestion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    databricks_environment = get_databricks_environment(spark_session)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    print("\n*** Running RefDB healthcheck ***")
    run_healthcheck_refdb(settings=settings)

    print("\n*** Running API healthcheck ***")
    run_healthcheck_api()

    print("\n*** Running SIA healthcheck ***")
    run_healthcheck_sia(settings=settings)


if __name__ == "__main__":
    main()
