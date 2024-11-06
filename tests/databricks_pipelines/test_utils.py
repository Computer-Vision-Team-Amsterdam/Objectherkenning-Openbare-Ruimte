import unittest

from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (
    unix_to_yyyy_mm_dd,
)


class TestUnixToYMDConversion(unittest.TestCase):
    def test_unix_to_yyyy_mm_dd(self):
        unix_timestamp = 1724668744
        expected_date = "2024-08-26"
        result = unix_to_yyyy_mm_dd(unix_timestamp)
        self.assertEqual(result, expected_date)


if __name__ == "__main__":
    unittest.main()
