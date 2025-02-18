import numpy as np
from databricks.sdk.runtime import sqlContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, mean, monotonically_increasing_id, row_number
from sklearn.cluster import DBSCAN

MS_PER_RAD = 6371008.8  # Earth radius in meters
MIN_SAMPLES = 1  # avoid noise points. All points are either in a cluster or are a cluster of their own


class Clustering:
    def __init__(self, spark: SparkSession, catalog, schema, detections, frames, 
                 active_object_categories):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.detection_metadata = detections
        self.frame_metadata = frames
        self.active_object_categories = active_object_categories
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

        # Only keep active objects
        joined_df = joined_df.where(col("object_class").
                                    isin(list(self.active_object_categories.keys())))

        return joined_df

    def _extract_containers_coordinates_with_detection_id(self):

        containers_df = self.joined_metadata.select(
            "detection_id", "gps_lat", "gps_lon", "object_class"
        )

        return containers_df

    def _cluster_points(self, eps, min_samples=MIN_SAMPLES):
        """
        Cluster points in a DataFrame using DBSCAN.

        Parameters:
        points (np.ndarray): [lon, lat] coordinates for each point
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        """
        # Get distinct object categories
        object_categories = [row["object_class"] for row in self.joined_metadata.select(
            "object_class").distinct().collect()]
        
        new_dfs = []
        global_offset = 0

        # Apply clustering per object category
        for cat in object_categories:
            df_cat = self.joined_metadata.filter(col("object_class") == cat)
            coords = np.array(
                df_cat.select("gps_lat", "gps_lon")
                .rdd.map(lambda row: (row["gps_lat"], row["gps_lon"]))
                .collect()
            )
            # Skip if there are no valid coordinates
            if coords.size == 0 or coords.ndim != 2 or coords.shape[1] != 2:
                continue
            
            db = DBSCAN(
                eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
            ).fit(np.radians(coords))
            
            # Build a local mapping from each unique raw label to a new unique tracking_id.
            raw_labels = [int(v) for v in db.labels_]
            unique_labels = sorted(set(raw_labels))
            local_mapping = {}
            for label in unique_labels:
                local_mapping[label] = global_offset
                global_offset += 1

            # Remap the raw labels using the local mapping.
            new_labels = [local_mapping[label] for label in raw_labels]

            # Add the new tracking_id column to this category's DataFrame.
            df_cat = self.add_column_to_df(df_cat, "tracking_id", new_labels)
            new_dfs.append(df_cat)
        
        if new_dfs:
            # Union all per-category clustered DataFrames into joined_metadata.
            self.joined_metadata = new_dfs[0]
            for df in new_dfs[1:]:
                self.joined_metadata = self.joined_metadata.union(df)
        else:
            print("No data to cluster after filtering.")


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
