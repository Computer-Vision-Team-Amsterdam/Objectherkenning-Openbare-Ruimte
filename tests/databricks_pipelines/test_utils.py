import unittest
from datetime import datetime

from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (
    get_landingzone_folder_for_timestamp,
)


class TestTimestampToFolderConversion(unittest.TestCase):
    def test_landingzone_from_timestamp(self):
        unix_timestamp = datetime.fromtimestamp(1724668744)
        expected_date = "2024-08-26"
        result = get_landingzone_folder_for_timestamp(unix_timestamp)
        self.assertEqual(result, expected_date)


if __name__ == "__main__":
    unittest.main()
