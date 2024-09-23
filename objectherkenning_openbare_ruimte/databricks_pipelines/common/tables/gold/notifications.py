from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class GoldSignalNotificationsManager(TableManager):
    table_name: str = "gold_signal_notifications"

    @staticmethod
    def get_table_name() -> str:
        return GoldSignalNotificationsManager.table_name
