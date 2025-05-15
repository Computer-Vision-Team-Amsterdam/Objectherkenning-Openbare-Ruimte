# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    get_databricks_environment,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.data_enrichment import (  # noqa: E402
    DataEnrichment,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def main():
    sparkSession = SparkSession.builder.appName("DataEnrichment").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)

    data_enrichment = DataEnrichment(
        sparkSession=sparkSession, catalog=catalog, schema=schema, settings=settings
    )
    data_enrichment.run_data_enrichment_step()


if __name__ == "__main__":
    main()
