from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeSignalNotificationsFeedbackManager(TableManager):
    table_name: str = "bronze_signal_notifications_feedback"

    @staticmethod
    def get_table_name() -> str:
        return BronzeSignalNotificationsFeedbackManager.table_name
