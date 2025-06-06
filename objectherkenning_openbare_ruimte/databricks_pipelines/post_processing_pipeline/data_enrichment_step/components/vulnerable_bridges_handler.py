from typing import List

import geopy.distance
from osgeo import osr  # pylint: disable-all
from pyspark.sql import Row, SparkSession
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.wkt import dumps as wkt_dumps


class VulnerableBridgesHandler:
    def __init__(
        self, spark_session: SparkSession, root_source, vuln_bridges_relative_path
    ):
        self.spark_session = spark_session
        self.file_path = f"{root_source}/{vuln_bridges_relative_path}"
        self._bridges_coordinates: List[List[List[float]]] = []
        self._bridges_ids: List[int] = []
        self._parse_bridges_coordinates()
        self._bridges_coordinates_geometry = self._convert_coordinates_to_linestring()

    def get_bridges_coordinates(self):
        return self._bridges_coordinates

    def get_bridges_coordinates_geometry(self):
        return self._bridges_coordinates_geometry

    def get_bridges_ids(self):
        return self._bridges_ids

    def _rd_to_wgs(self, coordinates: List[float]) -> List[float]:
        """
        Convert rijksdriehoekcoordinates into WGS84 cooridnates. Input parameters: x (float), y (float).
        """
        epsg28992 = osr.SpatialReference()
        epsg28992.ImportFromEPSG(28992)

        epsg4326 = osr.SpatialReference()
        epsg4326.ImportFromEPSG(4326)

        rd2latlon = osr.CoordinateTransformation(epsg28992, epsg4326)
        lonlatz = rd2latlon.TransformPoint(coordinates[0], coordinates[1])
        return [float(value) for value in lonlatz[:2]]

    def _parse_bridges_coordinates(self):
        """
        Creates a list of coordinates where to find vulnerable bridges and canal walls
        Coordinates of one bridge is a list of coordinates.

        [[[52.370292666750956, 4.868173855056686],
         [52.36974277094149, 4.868559544639536],
         [52.369742761981286, 4.868559550924023]
         ],
         [
             list of coordinates for second bridges
         ],
         ...
        ]
        """
        # Load the GeoJSON file into a Spark DataFrame
        sparkDataFrame = (
            self.spark_session.read.format("json")
            .option("driverName", "GeoJSON")
            .load(self.file_path)
        )

        # Filter out rows where the "geometry" column is not null
        filtered_df = sparkDataFrame.filter(sparkDataFrame.geometry.isNotNull())

        # Iterate through each feature in the DataFrame
        for feature_id, feature in enumerate(filtered_df.collect()):
            bridge_coords = []
            if feature["geometry"]["coordinates"]:
                for _, coords in enumerate(feature["geometry"]["coordinates"][0]):
                    bridge_coords.append(self._rd_to_wgs(coords))
                self._bridges_coordinates.append(bridge_coords)
                self._bridges_ids.append(feature_id)

    def _convert_coordinates_to_linestring(self):
        """
        We need the bridges coordinates as LineString to perform distance calculations
        """
        bridges_coordinates_geom = [
            LineString(bridge_coordinates)
            for bridge_coordinates in self._bridges_coordinates
            if bridge_coordinates
        ]
        return bridges_coordinates_geom

    @staticmethod
    def _line_to_point_in_meters(line: LineString, point: Point) -> float:
        """
        Calculates the shortest distance between a line and point, returns a float in meters
        """
        closest_point = nearest_points(line, point)[0]
        closest_point_in_meters = float(
            geopy.distance.distance(closest_point.coords, point.coords).meters
        )
        return closest_point_in_meters

    def calculate_distances_to_closest_vulnerable_bridges(
        self,
        bridges_locations_as_linestrings: List[LineString],
        objects_coordinates_df,
        bridges_ids: List[int],
        bridges_coordinates: List[List[List[float]]],
    ):
        results = []

        for row in objects_coordinates_df.collect():
            object_lat = row.gps_lat
            object_lon = row.gps_lon

            object_location = Point(object_lat, object_lon)
            bridge_object_distances = []
            for idx, bridge_location in enumerate(bridges_locations_as_linestrings):
                try:
                    # calculate distance between object point and bridge linestring
                    bridge_dist = VulnerableBridgesHandler._line_to_point_in_meters(
                        bridge_location, object_location
                    )
                except Exception:
                    bridge_dist = 10000
                    print("Error occurred:")
                    print(
                        f"Container location: {object_location}, {object_location.coords}"
                    )
                    print(f"Bridge location: {bridge_location}")
                bridge_object_distances.append(
                    (
                        bridge_dist,
                        bridges_ids[idx],  # ID of the bridge
                        bridges_coordinates[idx][0],  # Coordinates of the bridge
                        bridge_location,  # Bridge linestring (in WKT format)
                    )
                )
            (
                closest_bridge_distance,
                closest_bridge_id,
                closest_bridge_coord,
                closest_bridge_linestring,
            ) = min(bridge_object_distances, key=lambda x: x[0])
            results.append(
                Row(
                    detection_id=row.detection_id,  # retain the detection id for joining
                    closest_bridge_distance=round(closest_bridge_distance, 2),
                    closest_bridge_id=closest_bridge_id,
                    closest_bridge_coordinates=closest_bridge_coord,
                    closest_bridge_linestring_wkt=wkt_dumps(closest_bridge_linestring),
                )
            )

        results_df = self.spark_session.createDataFrame(results)

        return results_df
