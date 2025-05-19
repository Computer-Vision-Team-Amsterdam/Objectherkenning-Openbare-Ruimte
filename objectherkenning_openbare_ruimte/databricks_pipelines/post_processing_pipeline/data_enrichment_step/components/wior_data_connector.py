from typing import Dict, List, Optional

import geopy.distance
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col
from shapely.geometry import Point

from objectherkenning_openbare_ruimte.databricks_pipelines.common.reference_db_connector import (
    ReferenceDatabaseConnector,
)


class WIORDataHandler(ReferenceDatabaseConnector):
    wior_table_name = "wior_wior"

    def __init__(
        self,
        spark: SparkSession,
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        db_port: int,
        object_classes: Dict[int, str],
        permit_mapping: Dict[int, List],
    ) -> None:
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark = spark
        self.object_classes = object_classes
        self.permit_mapping = permit_mapping

    def query_and_process_wior_permits(self, date_to_query: str) -> None:
        """
        Query and process WIOR permits for a given date.

        Args:
            date_to_query (str): Date to query in YYYY-MM-DD format
        """
        query = f"""
        SELECT *,
          geometrie <-> ST_Transform(ST_SetSRID(ST_MakePoint(4.9041, 52.3676), 4326), 28992) AS afstand
        FROM {self.wior_table_name}
        WHERE '{date_to_query}' BETWEEN datum_start_uitvoering AND datum_einde_uitvoering
        ORDER BY afstand
        """  # nosec B608
        print(f"Querying the WIOR database for date {date_to_query}...")
        result_df = self.run(query)

        # Store rows where permit has valid coordinates as healthy data
        self._healthy_df = result_df[result_df["geometrie"].notnull()]

        # Reset the healthy DataFrame index
        self._healthy_df = self._healthy_df.reset_index(drop=True)

        print(f"{len(self._healthy_df)} WIOR permits found with valid coordinates.")
        if len(self._healthy_df) == 0:
            raise ValueError("No WIOR permits found for the given date.")

        # Store rows where permit has invalid coordinates as quarantine data
        self._quarantine_df = result_df[result_df["geometrie"].isnull()]
        print(
            f"{len(self._quarantine_df)} WIOR permits found with invalid coordinates."
        )

        self._permits_coordinates = self._extract_permits_coordinates()
        self._permits_coordinates_geometry = self._convert_coordinates_to_point()

    def _extract_permits_coordinates(self) -> List[tuple]:
        """
        Extract permit coordinates from the healthy DataFrame.

        Returns:
            List[tuple]: List of (latitude, longitude) tuples
        """
        permits_coordinates = list(
            self._healthy_df[["permit_lat", "permit_lon"]].itertuples(
                index=False, name=None
            )
        )
        return permits_coordinates

    def get_permits_ids(self) -> List[str]:
        """
        Get list of permit IDs from the healthy DataFrame.

        Returns:
            List[str]: List of permit IDs
        """
        permits_ids = self._healthy_df["id"].tolist()
        return permits_ids

    def _convert_coordinates_to_point(self) -> List[Point]:
        """
        Convert permit coordinates to Shapely Point objects for distance calculations.

        Returns:
            List[Point]: List of Shapely Point objects
        """
        permits_coordinates_geometry = [
            Point(location) for location in self._permits_coordinates
        ]
        return permits_coordinates_geometry

    def get_healthy_df(self) -> pd.DataFrame:
        """
        Get the healthy DataFrame containing valid permits.

        Returns:
            pd.DataFrame: DataFrame with valid permits
        """
        return self._healthy_df

    def get_quarantine_df(self) -> pd.DataFrame:
        """
        Get the quarantine DataFrame containing invalid permits.

        Returns:
            pd.DataFrame: DataFrame with invalid permits
        """
        return self._quarantine_df

    def calculate_distances_to_closest_permits(
        self, objects_coordinates_df: DataFrame
    ) -> DataFrame:
        """
        Calculate distances to the closest permits for each object class and union the results.

        Parameters:
            objects_coordinates_df (DataFrame): A Spark DataFrame containing object coordinates.

        Returns:
            DataFrame: A Spark DataFrame with calculated distances or an empty DataFrame if no results.
        """
        dfs_closest_permits: List[DataFrame] = []

        # Loop through each target object category
        for object_class in set(self.object_classes.keys()):
            df_object_class = objects_coordinates_df.filter(
                col("object_class") == object_class
            )
            if not df_object_class.take(1):
                continue

            df_closest_permit = (
                self.calculate_distances_to_closest_permits_for_object_class(
                    df_object_class, object_class
                )
            )
            if df_closest_permit:
                dfs_closest_permits.append(df_closest_permit)

        if dfs_closest_permits:
            union_dfs_closest_permits = dfs_closest_permits[0]
            for df in dfs_closest_permits[1:]:
                union_dfs_closest_permits = union_dfs_closest_permits.union(df)
            return union_dfs_closest_permits
        else:
            # Return an empty Spark DataFrame with the same schema as objects_coordinates_df
            empty_rdd = self.spark.sparkContext.emptyRDD()
            return self.spark.createDataFrame(
                empty_rdd, schema=objects_coordinates_df.schema
            )

    def calculate_distances_to_closest_permits_for_object_class(
        self, objects_df: DataFrame, category: int
    ) -> Optional[DataFrame]:
        """
        Calculate the closest permit distances for a given object category.

        Parameters:
            objects_df (DataFrame): Spark DataFrame with object data including 'gps_lat', 'gps_lon', and 'detection_id'.
            category (int): The target object category used to filter healthy permits.

        Returns:
            Optional[DataFrame]: A Spark DataFrame with columns 'detection_id', 'closest_permit_distance',
                                'closest_permit_id', 'closest_permit_lat', and 'closest_permit_lon'.
                                Returns None if no healthy permits match the category.
        """
        if not self._permits_coordinates_geometry:
            return None

        results = []
        for row in objects_df.collect():
            object_lat = row.gps_lat
            object_lon = row.gps_lon

            object_location = Point(object_lat, object_lon)

            closest_permit_distances = []
            for permit_location in self._permits_coordinates_geometry:
                try:
                    # Calculate distance between object point and permit point
                    permit_dist = geopy.distance.distance(
                        object_location.coords, permit_location.coords
                    ).meters
                except Exception:
                    permit_dist = 0
                    print("Error occurred:")
                    print(
                        f"Object location: {object_location}, {object_location.coords}"
                    )
                    print(
                        f"Permit location: {permit_location}, {permit_location.coords}"
                    )
                closest_permit_distances.append(permit_dist)

            min_distance_idx = np.argmin(closest_permit_distances)
            results.append(
                Row(
                    detection_id=row.detection_id,  # retain the detection id for joining
                    closest_permit_distance=float(
                        closest_permit_distances[min_distance_idx]
                    ),
                    closest_permit_id=self.get_permits_ids()[min_distance_idx],
                    closest_permit_lat=self._permits_coordinates[min_distance_idx][0],
                    closest_permit_lon=self._permits_coordinates[min_distance_idx][1],
                )
            )
        results_df = self.spark.createDataFrame(results)

        return results_df
