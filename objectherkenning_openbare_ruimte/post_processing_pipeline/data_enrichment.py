import os
import geojson
from tqdm import tqdm
from typing import List, Tuple, Union
from osgeo import osr  # pylint: disable-all


class BridgesCoordinatesParser:
    def __init__(self, file_path):
        self.file_path = file_path


    def rd_to_wgs(self, coordinates: List[float]) -> List[float]:
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

    def parse_bridges_coordinates(self) -> List[List[List[float]]]:
        """
        Return a list of coordinates where to find vulnerable bridges and canal walls
        """
        # Load the GeoJSON file into a Spark DataFrame
        sparkDataFrame = spark.read.format("json") \
            .option("driverName", "GeoJSON") \
            .load(self.file_path)

        # Filter out rows where the "geometry" column is not null
        filtered_df = sparkDataFrame.filter(sparkDataFrame.geometry.isNotNull())

        # Create an empty list to store bridge coordinates
        bridges_coords = []

        # Iterate through each feature in the DataFrame
        for feature in tqdm(filtered_df.collect(), desc="Parsing the bridges information"):
            bridge_coords = []
            if feature["geometry"]["coordinates"]:
                for idx, coords in enumerate(feature["geometry"]["coordinates"][0]):
                    bridge_coords.append(self.rd_to_wgs(coords))
                bridges_coords.append(bridge_coords)

        return bridges_coords


if __name__ == "__main__":
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"

    parser = BridgesCoordinatesParser(file_path)
    bridges = parser.parse_bridges_coordinates()
