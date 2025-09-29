# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    get_databricks_environment,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.delete_images_step.components.delete_images_step import (  # noqa: E402
    DeleteImagesStep,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def main() -> None:
    spark_session = SparkSession.builder.appName("ImageDeletion").getOrCreate()
    databricks_environment = get_databricks_environment(spark_session)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)

    deleteImagesStep = DeleteImagesStep(spark_session, catalog, schema, settings)
    deleteImagesStep.run()


if __name__ == "__main__":
    main()
