from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from shapely.geometry import Point
from pyspark.sql import SparkSession

from .databricks_workspace import get_catalog_name

class Clustering:

    def __init__(self, spark: SparkSession, date):
        
        self.spark = spark
        self.catalog = get_catalog_name(spark=spark)
        self.schema = "oor"
        self.detection_metadata = self.spark.read.table(f'{self.catalog}.{self.schema}.bronze_detection_metadata') # TODO change table to silver_detection_metadata after implementing metadata healthcheck
        self.frame_metadata = self.spark.read.table(f'{self.catalog}.{self.schema}.bronze_frame_metadata') # TODO change table to silver_detection_metadata after implementing metadata healthcheck
        self._filter_objects_by_date(date)
        self._filter_objects_randomly()
        self.df_joined= self._join_frame_and_detection_metadata()

        self._containers_coordinates = self._extract_containers_coordinates()  
        self._containers_coordinates_geometry = self._convert_coordinates_to_point()

    def get_containers_coordinates(self):
        return self._containers_coordinates
    
    def get_containers_coordinates_geometry(self):
        return self._containers_coordinates_geometry

    def _filter_objects_by_date(self, date):
        self.detection_metadata = self.detection_metadata.filter(self.detection_metadata["image_name"].like(f"%{date}%"))

    def _filter_objects_randomly(self):
        self.detection_metadata = self.detection_metadata.sample(False, 0.1, seed=42)  # 10% sample size

    def _filter_objects_by_tracking_id(self):
        pass   

    def _join_frame_and_detection_metadata(self):

        joined_df = self.frame_metadata.join(self.detection_metadata, self.frame_metadata["image_name"] == self.detection_metadata["image_name"])
        filtered_df = joined_df.filter(self.frame_metadata["image_name"] == self.detection_metadata["image_name"])
        columns = ["gps_date", "id", "object_class", "gps_lat", "gps_lon"]
        selected_df = filtered_df.select(columns)
        joined_df = selected_df \
                    .withColumnRenamed("gps_date", "detection_date") \
                    .withColumnRenamed("id", "detection_id")

        return joined_df
    
    def _extract_containers_coordinates(self):

        # Collect the DataFrame rows as a list of Row objects
        rows = self.df_joined.select("gps_lat", "gps_lon").collect()
    
        # Convert the list of Row objects into a list of tuples
        containers_coordinates = [(row['gps_lat'], row['gps_lon']) for row in rows]

        return containers_coordinates
           
    def _convert_coordinates_to_point(self):
        """
        We need the containers coordinates as Point to perform distance calculations
        """
        containers_coordinates_geometry = [Point(location) for location in self._containers_coordinates] 
        return containers_coordinates_geometry

    # why is this operation so complicated!?    
    def add_column(self, column_name, values):

        # Ensure the length of the values matches the number of rows in the dataframe
        if len(values) != self.df_joined.count():
            raise ValueError("The length of the list does not match the number of rows in the dataframe")

        #convert list to a dataframe    
        b = sqlContext.createDataFrame([(v,) for v in values], [column_name])

        #add 'sequential' index and join both dataframe to get the final result
        self.df_joined = self.df_joined.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
        b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

        self.df_joined = self.df_joined.join(b, self.df_joined.row_idx == b.row_idx).\
                    drop("row_idx")