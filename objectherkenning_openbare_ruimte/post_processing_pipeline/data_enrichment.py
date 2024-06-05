import os
import geojson
from tqdm import tqdm
from typing import List, Tuple, Union
from osgeo import osr  # pylint: disable-all


def get_bridge_information(file: str) -> List[List[List[float]]]:
    """
    Return a list of coordinates where to find vulnerable bridges and canal walls
    """

    def rd_to_wgs(coordinates: List[float]) -> List[float]:
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

    bridges_coords = []
    with open(file) as f:
        gj = geojson.load(f)
    features = gj["features"]
    print("Parsing the bridges information")
    for feature in tqdm(features, total=len(features)):
        bridge_coords = []
        if feature["geometry"]["coordinates"]:
            for idx, coords in enumerate(feature["geometry"]["coordinates"][0]):
                bridge_coords.append(rd_to_wgs(coords))
            # only add to the list when there are coordinates
            bridges_coords.append(bridge_coords)
    return bridges_coords



if __name__ == "__main__":
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net/"
    vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"

    sparkDataFrame = spark.read.format("json")\
                    .option("driverName", "GeoJSON")\
                    .load(file_path)
    
    # display content
    sparkDataFrame.display()

    
    #bridges = get_bridge_information(file_path)