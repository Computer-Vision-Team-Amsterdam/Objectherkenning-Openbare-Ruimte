from functools import reduce
from typing import Dict

import numpy as np
from databricks.sdk.runtime import sqlContext
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, mean, monotonically_increasing_id, row_number
from sklearn.cluster import DBSCAN

MS_PER_RAD = 6371008.8  # Earth radius in meters
MIN_SAMPLES = 1  # avoid noise points. All points are either in a cluster or are a cluster of their own


class Clustering:

    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        detections: DataFrame,
        frames: DataFrame,
        active_object_classes: Dict[int, str],
    ) -> None:
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.detection_metadata = detections
        self.frame_metadata = frames
        self.active_object_classes = active_object_classes
        self.joined_metadata = self._join_frame_and_detection_metadata()
        self._containers_coordinates_with_detection_id = None

        self.filter_by_confidence_score(0.8)
        self.filter_by_bounding_box_size(0.003)

        if self.detection_metadata.count() == 0 or self.frame_metadata.count() == 0:
            print("Missing or incomplete data to run clustering. Stopping execution.")
            return

        self.cluster_and_select_images()

    def filter_by_confidence_score(self, min_conf_score: float):
        self.joined_metadata = self.joined_metadata.where(
            col("confidence") > min_conf_score
        )

    def filter_by_bounding_box_size(self, min_bbox_size: float):
        # Calculate area for each image
        self.joined_metadata = self.joined_metadata.withColumn(
            "area", col("width") * col("height")
        )
        self.joined_metadata = self.joined_metadata.where(col("area") > min_bbox_size)

    def get_containers_coordinates_with_detection_id(self):
        if not self._containers_coordinates_with_detection_id:
            self._containers_coordinates_with_detection_id = (
                self._extract_containers_coordinates_with_detection_id()
            )
        return self._containers_coordinates_with_detection_id

    def _join_frame_and_detection_metadata(self):
        """
        Join frame and detection metadata, keeping only active objects.

        Returns
        -------
        DataFrame
            A Spark DataFrame containing joined metadata with the following columns:
            'detection_date', 'detection_id', 'object_class', 'image_name', 'x_center', 'y_center',
            'width', 'height', 'confidence', 'gps_lat', 'gps_lon'.
        """
        active_object_classes = list(self.active_object_classes.keys())
        # Filter detection metadata for active object classes before joining.
        active_detection_metadata = self.detection_metadata.where(
            col("object_class").isin(active_object_classes)
        )

        joined_df = self.frame_metadata.alias("fm").join(
            active_detection_metadata.alias("dm"),
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

        containers_df = self.joined_metadata.select(
            "detection_id", "gps_lat", "gps_lon", "object_class"
        )

        return containers_df

    def _cluster_points(self, eps):
        """
        Cluster points in a DataFrame using DBSCAN across different object classes.

        Parameters:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        """
        # Get distinct object classes
        detected_classes = [
            row["object_class"]
            for row in self.joined_metadata.select("object_class").distinct().collect()
        ]

        dfs_clustered = []
        cluster_id_counter = 0

        # Apply clustering per object class
        for detected_class in detected_classes:
            df_class = self.joined_metadata.filter(
                col("object_class") == detected_class
            )
            df_clustered, cluster_id_counter = self._cluster_points_for_class(
                df_class, eps, cluster_id_counter, min_samples=MIN_SAMPLES
            )
            if df_clustered is not None:
                dfs_clustered.append(df_clustered)

        if dfs_clustered:
            # Union all per-category clustered DataFrames using reduce.
            self.joined_metadata = reduce(
                lambda df1, df2: df1.union(df2), dfs_clustered
            )
        else:
            print("No data to cluster after filtering.")

    def _cluster_points_for_class(
        self, df_metadata_by_class, eps, min_samples, cluster_id_counter
    ):
        """
        Perform DBSCAN clustering on a single class DataFrame.

        Parameters:
            df_metadata_by_class (DataFrame): DataFrame filtered for one object class.
            eps (float): Maximum distance between two samples for them to be considered neighbors.
            min_samples (int): Minimum number of samples for a point to be considered a core point.
            cluster_id_counter (int): Current counter to assign unique tracking IDs.

        Returns:
            tuple: (clustered DataFrame with new 'tracking_id' column, updated cluster_id_counter)
        """
        # Extract coordinates as a numpy array.
        coordinates = np.array(
            df_metadata_by_class.select("gps_lat", "gps_lon")
            .rdd.map(lambda row: (row["gps_lat"], row["gps_lon"]))
            .collect()
        )

        # Skip if there are no valid coordinates.
        if coordinates.size == 0 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
            return None, cluster_id_counter

        # Apply DBSCAN clustering.
        db = DBSCAN(
            eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
        ).fit(np.radians(coordinates))

        # Remap raw labels to new unique tracking IDs using dictionary comprehension.
        raw_labels = [int(v) for v in db.labels_]
        unique_labels = sorted(set(raw_labels))
        local_mapping = {
            label: cluster_id_counter + i for i, label in enumerate(unique_labels)
        }
        cluster_id_counter += len(unique_labels)

        new_labels = [local_mapping[label] for label in raw_labels]

        # Add the new 'tracking_id' column to this category's DataFrame.
        df_clustered = self.add_column_to_df(
            df_metadata_by_class, "tracking_id", new_labels
        )
        return df_clustered, cluster_id_counter

    def cluster_and_select_images(self, distance=10):
        # Cluster the points based on distance
        epsilon = distance / MS_PER_RAD  # radius of the neighborhood
        self._cluster_points(eps=epsilon)

        # Calculate the mean confidence for each cluster
        window_spec = Window.partitionBy("tracking_id")
        self.joined_metadata = self.joined_metadata.withColumn(
            "mean_confidence", mean("confidence").over(window_spec)
        )

        # Select images with confidence above the mean confidence of their cluster
        self.joined_metadata = self.joined_metadata.filter(
            col("confidence") >= col("mean_confidence")
        )

        # Select the image with the largest area within each cluster
        window_spec_area = Window.partitionBy("tracking_id").orderBy(col("area").desc())
        self.joined_metadata = self.joined_metadata.withColumn(
            "row_number", row_number().over(window_spec_area)
        )
        self.joined_metadata = self.joined_metadata.filter(col("row_number") == 1).drop(
            "row_number", "mean_confidence", "area"
        )

        # Update container coordinates after clustering
        self._containers_coordinates_with_detection_id = (
            self._extract_containers_coordinates_with_detection_id()
        )

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
