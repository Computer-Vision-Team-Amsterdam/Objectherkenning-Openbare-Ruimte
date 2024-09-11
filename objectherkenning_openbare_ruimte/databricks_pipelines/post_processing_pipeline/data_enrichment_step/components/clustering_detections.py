import numpy as np
from databricks.sdk.runtime import sqlContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, mean, monotonically_increasing_id, row_number
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

MS_PER_RAD = 6371008.8  # Earth radius in meters
MIN_SAMPLES = 1  # avoid noise points. All points are either in a cluster or are a cluster of their own


class Clustering:
    def __init__(self, spark: SparkSession, catalog, schema):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        query_detection_metadata = f"SELECT * FROM {self.catalog}.{self.schema}.silver_detection_metadata WHERE status='Pending'"  # nosec
        self.detection_metadata = self.spark.sql(query_detection_metadata)
        print(
            f"03: Loaded {self.detection_metadata.count()} 'Pending' rows from {self.catalog}.oor.silver_detection_metadata."
        )

        query_frame_metadata = f"SELECT * FROM {self.catalog}.{self.schema}.silver_frame_metadata WHERE status='Pending'"  # nosec
        self.frame_metadata = self.spark.sql(query_frame_metadata)
        print(
            f"03: Loaded {self.frame_metadata.count()} 'Pending' rows from {self.catalog}.oor.silver_frame_metadata."
        )

        self.df_joined = self._join_frame_and_detection_metadata()
        self._containers_coordinates = None
        self._containers_coordinates_geometry = None

    def filter_by_confidence_score(self, min_conf_score: float):
        self.df_joined = self.df_joined.where(col("confidence") > min_conf_score)

    def filter_by_bounding_box_size(self, min_bbox_size: float):
        # Calculate area for each image
        self.df_joined = self.df_joined.withColumn("area", col("width") * col("height"))
        self.df_joined = self.df_joined.where(col("area") > min_bbox_size)

    def get_containers_coordinates(self):
        if not self._containers_coordinates:
            self._containers_coordinates = self._extract_containers_coordinates()
        return self._containers_coordinates

    def get_containers_coordinates_geometry(self):
        if not self._containers_coordinates_geometry:
            self._containers_coordinates_geometry = self._convert_coordinates_to_point()
        return self._containers_coordinates_geometry

    def _filter_objects_by_date(self, date):
        self.detection_metadata = self.detection_metadata.filter(
            self.detection_metadata["image_name"].like(f"%{date}%")
        )

    def _filter_objects_randomly(self):
        self.detection_metadata = self.detection_metadata.sample(
            False, 0.1, seed=42
        )  # 10% sample size

    def _filter_objects_by_tracking_id(self):
        pass

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
            # col("fm.gps_lat"),
            # col("fm.gps_lon"),
            col("fm.gps_lat").cast("float"),
            col("fm.gps_lon").cast("float"),
        ]

        joined_df = joined_df.select(columns)

        return joined_df

    def _extract_containers_coordinates(self):
        # Collect the DataFrame rows as a list of Row objects
        rows = self.df_joined.select("gps_lat", "gps_lon").collect()
        # Convert the list of Row objects into a list of tuples
        containers_coordinates = [(row["gps_lat"], row["gps_lon"]) for row in rows]

        return containers_coordinates

    def _convert_coordinates_to_point(self):
        """
        We need the containers coordinates as Point to perform distance calculations
        """
        containers_coordinates_geometry = [
            Point(location) for location in self.get_containers_coordinates()
        ]
        return containers_coordinates_geometry

    # why is this operation so complicated!?
    def add_column(self, column_name, values):

        # Ensure the length of the values matches the number of rows in the dataframe
        if len(values) != self.df_joined.count():
            raise ValueError(
                "The length of the list does not match the number of rows in the dataframe"
            )

        # convert list to a dataframe
        b = sqlContext.createDataFrame(
            [(v,) for v in values], [column_name]
        )  # noqa: F405

        # add 'sequential' index and join both dataframe to get the final result
        self.df_joined = self.df_joined.withColumn(
            "row_idx", row_number().over(Window.orderBy(monotonically_increasing_id()))
        )
        b = b.withColumn(
            "row_idx", row_number().over(Window.orderBy(monotonically_increasing_id()))
        )

        self.df_joined = self.df_joined.join(
            b, self.df_joined.row_idx == b.row_idx
        ).drop("row_idx")

    def add_columns(self, columns_dict):
        for column_name, values in columns_dict.items():
            self.add_column(column_name, values)

    def _cluster_points(self, eps, min_samples=MIN_SAMPLES):
        """
        Cluster points in a DataFrame using DBSCAN.

        Parameters:
        points (np.ndarray): [lon, lat] coordinates for each point
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        """

        coordinates = np.array(self.get_containers_coordinates())

        db = DBSCAN(
            eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
        ).fit(np.radians(coordinates))

        # Add cluster labels to the DataFrame
        labels = [int(v) for v in db.labels_]
        self.add_column("tracking_id", labels)

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
        self._containers_coordinates = self._extract_containers_coordinates()
        self._containers_coordinates_geometry = self._convert_coordinates_to_point()

    def setup(self):
        self.filter_by_confidence_score(0.7)
        self.filter_by_bounding_box_size(0.003)

        if self.detection_metadata.count() == 0 or self.frame_metadata.count() == 0:
            print(
                "03: Missing or incomplete data to run clustering. Stopping execution."
            )
            return

        self.cluster_and_select_images()
