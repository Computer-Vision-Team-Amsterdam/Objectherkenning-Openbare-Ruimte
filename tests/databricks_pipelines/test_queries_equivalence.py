import unittest

# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.objects import (  # noqa: E402
    SilverObjectsPerDayManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    compare_dataframes,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


class TestSilverObjectsPerDayManager(unittest.TestCase):

    def __init__(self):
        self.sparkSession = SparkSession.builder.appName("TestCase").getOrCreate()
        databricks_environment = get_databricks_environment(self.sparkSession)
        project_root = os.path.dirname(os.path.dirname(os.getcwd()))
        config_file_path = os.path.join(project_root, "config_db.yml")
        self.settings = load_settings(config_file_path)["databricks_pipelines"][
            f"{databricks_environment}"
        ]

    def test_get_top_pending_records(self):
        silverObjectsPerDayManager = SilverObjectsPerDayManager(
            spark=self.sparkSession,
            catalog=self.settings["catalog"],
            schema=self.settings["schema"],
        )
        query = f"""
            SELECT * FROM {self.catalog}.{self.schema}.{silverObjectsPerDayManager.get_table_name()}
            WHERE status = 'Pending' AND object_class = 2 AND score >= 0.4
            ORDER BY score DESC
            LIMIT 20
            """  # nosec

        top_scores_df = self.sparkSession.sql(query)
        top_scores_df_no_sql = (
            silverObjectsPerDayManager.get_top_pending_records_no_sql(limit=20)
        )

        compare_dataframes(
            top_scores_df,
            top_scores_df_no_sql,
            df1_name="SQL filtered df",
            df2_name="PySpark filtered df",
        )


if __name__ == "__main__":
    unittest.main()
