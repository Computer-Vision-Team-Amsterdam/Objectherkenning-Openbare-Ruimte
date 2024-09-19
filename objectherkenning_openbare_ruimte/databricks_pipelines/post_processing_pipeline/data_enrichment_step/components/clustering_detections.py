import numpy as np
from databricks.sdk.runtime import sqlContext
from pyspark.sql import Row, SparkSession, Window
from pyspark.sql.functions import col, mean, monotonically_increasing_id, row_number
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

MS_PER_RAD = 6371008.8  # Earth radius in meters
MIN_SAMPLES = 1  # avoid noise points. All points are either in a cluster or are a cluster of their own


class Clustering:
    def __init__(self, spark: SparkSession, catalog, schema, detections, frames):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.detection_metadata = detections
        self.frame_metadata = frames
        self.df_joined = self._join_frame_and_detection_metadata()
        self._containers_coordinates_with_detection_id = None
        self._containers_coordinates_with_detection_id_and_geometry = None

    def filter_by_confidence_score(self, min_conf_score: float):
        self.df_joined = self.df_joined.where(col("confidence") > min_conf_score)

    def filter_by_bounding_box_size(self, min_bbox_size: float):
        # Calculate area for each image
        self.df_joined = self.df_joined.withColumn("area", col("width") * col("height"))
        self.df_joined = self.df_joined.where(col("area") > min_bbox_size)

    def get_containers_coordinates_with_detection_id(self):
        if not self._containers_coordinates_with_detection_id:
            self._containers_coordinates_with_detection_id = (
                self._extract_containers_coordinates_with_detection_id()
            )
        return self._containers_coordinates_with_detection_id

    def get_containers_coordinates_with_detection_id_and_geometry(self):
        if not self._containers_coordinates_with_detection_id_and_geometry:
            self._containers_coordinates_with_detection_id_and_geometry = (
                self.add_geometry_to_containers_coordinates()
            )
        return self._containers_coordinates_with_detection_id_and_geometry

    def _join_frame_and_detection_metadata(self):

        joined_df = self.frame_metadata.alias("fm").join(
            self.detection_metadata.alias("dm"),
            col("fm.image_name") == col("dm.image_name"),
        )

        columns = [
            col("fm.gps_date").alias("detection_date"),
            col("dm.id").alias("detection_id"),
            col("dm.object_class"),
            col("dm.image_name"),
            col("dm.x_center"),
            col("dm.y_center"),
            col("dm.width"),
            col("dm.height"),
            col("dm.confidence"),
            col("fm.gps_lat").cast("float"),
            col("fm.gps_lon").cast("float"),
        ]

        joined_df = joined_df.select(columns)

        return joined_df

    def _extract_containers_coordinates_with_detection_id(self):

        containers_df = self.df_joined.select("detection_id", "gps_lat", "gps_lon")

        return containers_df

    def add_geometry_to_containers_coordinates(self):
        """
        We need the containers coordinates as Point to perform distance calculations
        """
        containers_df = self._extract_containers_coordinates_with_detection_id()

        containers_df_with_geom = containers_df.rdd.map(
            lambda row: Row(
                **row.asDict(),  # retain all original columns in the row
                geometry=Point(row["gps_lat"], row["gps_lon"])
            )
        ).toDF()

        return containers_df_with_geom

    def _cluster_points(self, eps, min_samples=MIN_SAMPLES):
        """
        Cluster points in a DataFrame using DBSCAN.

        Parameters:
        points (np.ndarray): [lon, lat] coordinates for each point
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        """

        containers_df = self.get_containers_coordinates_with_detection_id()
        coordinates = np.array(
            containers_df.select("gps_lat", "gps_lon")
            .rdd.map(lambda row: (row["gps_lat"], row["gps_lon"]))
            .collect()
        )

        db = DBSCAN(
            eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
        ).fit(np.radians(coordinates))

        # Add cluster labels to the DataFrame
        labels = [int(v) for v in db.labels_]
        self.df_joined = self.add_column_to_df(
            df=self.df_joined, column_name="tracking_id", values=labels
        )

    def cluster_and_select_images(self, distance=10):
        # Cluster the points based on distance
        epsilon = distance / MS_PER_RAD  # radius of the neighborhood
        self._cluster_points(eps=epsilon)

        # Calculate the mean confidence for each cluster
        window_spec = Window.partitionBy("tracking_id")
        self.df_joined = self.df_joined.withColumn(
            "mean_confidence", mean("confidence").over(window_spec)
        )

        # Select images with confidence above the mean confidence of their cluster
        self.df_joined = self.df_joined.filter(
            col("confidence") >= col("mean_confidence")
        )

        # Select the image with the largest area within each cluster
        window_spec_area = Window.partitionBy("tracking_id").orderBy(col("area").desc())
        self.df_joined = self.df_joined.withColumn(
            "row_number", row_number().over(window_spec_area)
        )
        self.df_joined = self.df_joined.filter(col("row_number") == 1).drop(
            "row_number", "mean_confidence", "area"
        )

        # Update container coordinates after clustering
        self._containers_coordinates_with_detection_id = (
            self._extract_containers_coordinates_with_detection_id()
        )
        self._containers_coordinates_with_detection_id_and_geometry = (
            self.add_geometry_to_containers_coordinates()
        )

    def setup(self):
        self.filter_by_confidence_score(0.7)
        self.filter_by_bounding_box_size(0.003)

        if self.detection_metadata.count() == 0 or self.frame_metadata.count() == 0:
            print(
                "03: Missing or incomplete data to run clustering. Stopping execution."
            )
            return

        self.cluster_and_select_images()

    def add_column_to_df(self, df, column_name, values):
        # Ensure the length of the values matches the number of rows in the dataframe
        if len(values) != df.count():
            raise ValueError(
                "The length of the list does not match the number of rows in the dataframe"
            )

        # convert list to a dataframe
        b = sqlContext.createDataFrame(
            [(v,) for v in values], [column_name]
        )  # noqa: F405

        # add 'sequential' index and join both dataframe to get the final result
        df = df.withColumn(
            "row_idx", row_number().over(Window.orderBy(monotonically_increasing_id()))
        )
        b = b.withColumn(
            "row_idx", row_number().over(Window.orderBy(monotonically_increasing_id()))
        )

        df = df.join(b, df.row_idx == b.row_idx).drop("row_idx")

        return df
