import os
import geojson
from tqdm import tqdm
from typing import List, Tuple, Union
from osgeo import osr  # pylint: disable-all
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from pyspark.sql import SparkSession

import geopy.distance

class VulnerableBridgesHandler:
    def __init__(self, spark: SparkSession, file_path):
        self.spark = spark
        self.file_path = file_path
        self._bridges_coordinates = []
        self._parse_bridges_coordinates()
        self._bridges_coordinates_geometry = self._convert_coordinates_to_linestring()

    def get_bridges_coordinates(self):
        return self._bridges_coordinates
    
    def get_bridges_coordinates_geometry(self):
        return self._bridges_coordinates_geometry

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

    def _parse_bridges_coordinates(self) -> List[List[List[float]]]:
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
        sparkDataFrame = self.spark.read.format("json") \
            .option("driverName", "GeoJSON") \
            .load(self.file_path)

        # Filter out rows where the "geometry" column is not null
        filtered_df = sparkDataFrame.filter(sparkDataFrame.geometry.isNotNull())

        # Iterate through each feature in the DataFrame
        for feature in tqdm(filtered_df.collect(), desc="Parsing the bridges information"):
            bridge_coords = []
            if feature["geometry"]["coordinates"]:
                for idx, coords in enumerate(feature["geometry"]["coordinates"][0]):
                    bridge_coords.append(self._rd_to_wgs(coords))
                self._bridges_coordinates.append(bridge_coords)
    
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
        closest_point_in_meters = float(geopy.distance.distance(closest_point.coords, point.coords).meters)
        return closest_point_in_meters

    @staticmethod
    def calculate_distances_to_closest_vulnerable_bridges(bridges_locations_as_linestrings: List[LineString], containers_locations_as_points: List[Point]):
        bridges_distances = []
        for container_location in containers_locations_as_points:
            bridge_container_distances = []
            for bridge_location in bridges_locations_as_linestrings:
                try:  
                    bridge_dist = VulnerableBridgesHandler._line_to_point_in_meters(bridge_location, container_location)
                except:
                    bridge_dist = 10000
                    print("Error occured:")
                    print(f"Container location: {container_location}, {container_location.coords}")
                    print(f"Bridge location: {bridge_location}")
                bridge_container_distances.append(bridge_dist)
            closest_bridge_distance = min(bridge_container_distances)
            bridges_distances.append(round(closest_bridge_distance, 2))    
        return bridges_distances 