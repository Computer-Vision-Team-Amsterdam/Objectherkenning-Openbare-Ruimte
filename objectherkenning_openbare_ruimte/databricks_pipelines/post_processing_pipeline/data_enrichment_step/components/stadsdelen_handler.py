from typing import Any, List, Optional, Tuple

import requests
from pyproj import Transformer
from pyspark.sql import DataFrame, Row, SparkSession
from shapely import Polygon
from shapely.geometry import Point


class StadsdelenHandler:
    """
    Query the API for stadsdelen geometries and process the results.
    """

    def __init__(self, spark_session: SparkSession) -> None:
        self.spark_session = spark_session
        self.detection_crs = "EPSG:4326"
        self.stadsdelen_crs = "EPSG:28992"
        self.transformer = Transformer.from_crs(
            self.detection_crs, self.stadsdelen_crs, always_xy=True
        )
        self.stadsdelen: List[dict[str, Any]] = []
        self._query_and_process_stadsdelen()

    def _query_and_process_stadsdelen(self) -> bool:
        """
        Query the API for stadsdelen geometries and process the results.

        The function runs a query to fetch geometries for stadsdelen and
        converts them to Shapely geometry objects.
        """
        success = False
        url = "https://api.data.amsterdam.nl/v1/gebieden/stadsdelen/"
        print("Querying stadsdelen API...")
        try:
            result = requests.get(url, timeout=5)
            result.raise_for_status()

            for stadsdeel in result.json()["_embedded"]["stadsdelen"]:
                self.stadsdelen.append(
                    {
                        "name": stadsdeel["naam"],
                        "code": stadsdeel["code"],
                        "polygon": Polygon(stadsdeel["geometrie"]["coordinates"][0]),
                    }
                )
            print(f"Query successful, {len(self.stadsdelen)} stadsdelen returned.")
            success = True
        except requests.exceptions.RequestException as e:
            print(f"Error querying stadsdelen API: {e}")
            raise e
        return success

    def _get_stadsdeel_name_and_code(
        self, lat_lon: Tuple[float, float]
    ) -> Optional[Tuple[str, str]]:
        """
        Get the stadsdeel name and code for the given detection location.

        Parameters:
            lat_lon: A Tuple (lat, lon) with teh GPS coordinates

        Returns:
            A tuple ("name", "code") if the point is inside a stadsdeel, or None otherwise
        """
        (object_lat, object_lon) = lat_lon
        x, y = self.transformer.transform(object_lon, object_lat)
        pt = Point(x, y)

        for stadsdeel in self.stadsdelen:
            if stadsdeel["polygon"].contains(pt):
                return (stadsdeel["name"], stadsdeel["code"])

        return None

    def lookup_stadsdeel_for_detections(
        self,
        objects_coordinates_df: DataFrame,
    ) -> DataFrame:
        """
        Look up the stadsdeel for each detection location. Returns a DataFrame
        with the columns `stadsdeel` and `stadsdeel_code` for each row in
        objects_coordinates_df.
        """
        results = []

        for row in objects_coordinates_df.collect():
            lat_lon = (row.gps_lat, row.gps_lon)
            result = self._get_stadsdeel_name_and_code(lat_lon)
            if result:
                results.append(
                    Row(
                        detection_id=row.detection_id,  # retain the detection id for joining
                        stadsdeel=result[0],
                        stadsdeel_code=result[1],
                    )
                )
            else:
                print(f"Object at location {lat_lon} not within a known Stadsdeel.")

        if results:
            results_df = self.spark_session.createDataFrame(results)
        else:
            # Return an empty Spark DataFrame with the same schema as objects_coordinates_df.
            empty_rdd = self.spark_session.sparkContext.emptyRDD()
            results_df = self.spark_session.createDataFrame(
                empty_rdd, schema=objects_coordinates_df.schema
            )

        return results_df
