import sys
import os
sys.path.append(os.path.abspath('..'))

from helpers.databricks_workspace import get_catalog_name

    
class Clustering:

    def __init__(self, date):
        
        self.catalog = get_catalog_name()
        self.schema = "oor"
        self.detection_metadata = self.park.read.table(f'{self.catalog}.{self.schema}.bronze_detection_metadata') # TODO change table to silver_detection_metadata after implementing metadata healthcheck
        self.frame_metadata = self.spark.read.table(f'{self.catalog}.{self.schema}.bronze_frame_metadata') # TODO change table to silver_detection_metadata after implementing metadata healthcheck
        self.filter_objects_by_date(date)
        self.filter_objects_randomly()

    def filter_objects_by_date(self, date):
        self.detection_metadata = self.detection_metadata.filter(self.detection_metadata["image"].like(f"%{date}%"))

    def filter_objects_randomly(self):
        self.detection_metadata = self.detection_metadata.sample(False, 0.1, seed=42)  # 10% sample size

    def filter_objects_by_tracking_id(self):
        pass   

    def combine_data(self):

        joined_df = self.frame_metadata.join(self.detection_metadata, self.frame_metadata["image"] == self.detection_metadata["image"])
        filtered_df = joined_df.filter(self.frame_metadata["image"] == self.detection_metadata["image"])
        columns = ["gps_date", "id", "object_class", "gps_lat", "gps_lon"]
        selected_df = filtered_df.select(columns)
        renamed_df = selected_df \
                    .withColumnRenamed("gps_date", "detection_date") \
                    .withColumnRenamed("id", "detection_id")

        return renamed_df
