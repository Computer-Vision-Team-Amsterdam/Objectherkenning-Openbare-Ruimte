# this fixes the caching issues
dbutils.library.restartPython() 

from pyspark.sql import SparkSession
from helpers.decos_data_connector import DecosDataHandler
from helpers.vulnerable_bridges_handler import VulnerableBridgesHandler

import folium
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.wkt import loads
from IPython.display import display, IFrame

def generate_map(
    vulnerable_bridges,
    permit_locations,
    trajectory = None,
    detections = None,
    name = None,
    colors = None,
) -> None:
    """
    This method generates an HTML page with a map containing a path line and randomly chosen points on the line
    corresponding to detected containers on the path.

    :param vulnerable_bridges: list of line string coordinates.
    :param permit_locations: list of point coordinates.
    :param trajectory: list of coordinates that define the path.
    :param detections: model predictions dict (with information about file names and coordinates).
    :param name: custom name for the map. If not passed, name is created based on what the map contains.
    :param colors: colors to be assigned to each cluster
    """
    # Amsterdam coordinates
    latitude = 52.377956
    longitude = 4.897070

    # create empty map zoomed on Amsterdam
    Map = folium.Map(location=[latitude, longitude], zoom_start=12)
    popup = None
    # add container locations to the map
    if detections:
        for point in detections:
            folium.Marker(
                location=[point.x, point.y], 
                color="blue",
                popup=popup if popup else None,
                radius=5,
            ).add_to(Map)

    vulnerable_bridges_group = folium.FeatureGroup(name="Vulnerable bridges").add_to(
        Map
    )

    # add data of vulnerable bridges and canal walls to the map
    for linestring in vulnerable_bridges:
        coordinates = [(point[0], point[1]) for point in linestring.coords]
        folium.PolyLine(coordinates, color="yellow", weight=5, opacity=0.8).add_to(vulnerable_bridges_group)

    # add permit locations on the map
    for point in permit_locations:
        folium.CircleMarker(
            location=[point.x, point.y], color="red", radius=5, weight=2
        ).add_to(Map)

    # Function to find the closest point on a linestring to a given point
    def closest_point_on_linestring(point, linestring):
        #print(f'Nearest_points: {nearest_points(point,linestring)}')
        return nearest_points(point, linestring)[1]
    
    # add distances between container and closest vulnerable bridge
    # add distances between container and closest permit
    distances_group = folium.FeatureGroup(name="Distances to Bridges").add_to(Map)
    permit_distances_group = folium.FeatureGroup(name="Distances to Permits").add_to(Map)

    for detection in detections:
        print(f'Detection: {detection}')
        min_distance = float('inf')
        closest_bridge = None
        closest_point = None

        for bridge in vulnerable_bridges:
            point_on_bridge = closest_point_on_linestring(detection, bridge)
            distance = detection.distance(point_on_bridge)
            if distance < min_distance:
                min_distance = distance
                closest_point = point_on_bridge
        print(f'min_distance is {min_distance}, closest_point is: {closest_point}')

        if closest_point is not None:
            polyline_coords = [(detection.x, detection.y), (closest_point.x, closest_point.y)]
            folium.PolyLine(polyline_coords, color="blue", weight=5, opacity=0.8).add_to(distances_group)
            print(f'Adding PolyLine([{detection},{closest_point}] to distances_group. ')

        min_distance_permit = float('inf')
        closest_permit = None

        for permit in permit_locations:
            distance = detection.distance(permit)
            if distance < min_distance_permit:
                min_distance_permit = distance
                closest_permit = permit

        if closest_permit is not None:
            polyline_coords = [(detection.x, detection.y), (closest_permit.x, closest_permit.y)]
            folium.PolyLine(polyline_coords, color="green", weight=5, opacity=0.8).add_to(permit_distances_group)
            print(f'Adding PolyLine from {detection} to {closest_permit}, distance: {min_distance_permit}')

    folium.LayerControl().add_to(Map)

    # create name for the map
    if not name:
        if detections and trajectory:
            name = "Daily trajectory and predicted containers"
        if detections and not trajectory:
            name = "Daily predicted containers"
        if not detections and trajectory:
            name = "Daily trajectory"
        if not detections and not trajectory:
            name = "Empty map"
    print(f"Map is saved at {name}")
    Map.save(f"{name}.html")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Example").getOrCreate()

<<<<<<< Updated upstream
=======
def decos():
    pass


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Example").getOrCreate()

>>>>>>> Stashed changes
    # Setup permit data
    az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
    db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
    db_name = "mdbdataservices"
 
    decosDataHandler = DecosDataHandler(spark, az_tenant_id, db_host, db_name, db_port=5432)
    query = "SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"
    decosDataHandler.run(query)
    query_result_df = decosDataHandler.get_query_result_df()
<<<<<<< Updated upstream
    decosDataHandler.process_query_result()
    healthy_df = decosDataHandler.get_healthy_df()
    coords_geom = decosDataHandler.get_permits_coordinates_geometry()

    # Setup bridges data
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"
    bridgesHandler = VulnerableBridgesHandler(spark, file_path)
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()

    # Hard-coded containers
    coords_containers = [
    (4.8980294, 52.3617859),
    (4.8984923, 52.3618679),
    (4.9020681, 52.362536),
    (4.902779, 52.3636647),
    (4.9031758, 52.3634548),
    (4.9035023, 52.3628395),
    (4.9040048, 52.3618644),
    (4.9040796, 52.36172),
    (4.904525, 52.3608553),
    (4.9047965, 52.360528),
    (4.9053949, 52.3601648),
    (4.9056244, 52.3596579),
    (4.9059358, 52.3590319)]
    coords_containers_points = [Point(x, y) for y, x in coords_containers]

    generate_map(
        vulnerable_bridges=bridges_coordinates_geometry,
        permit_locations=coords_geom,
        detections=coords_containers_points,
        name="Decos containers",
    )

    
=======
    display(query_result_df)
    # decosDataHandler.process_query_result()
    # healthy_df = decosDataHandler.get_healthy_df()
    # display(healthy_df)
    # coords_geom = decosDataHandler.get_permits_coordinates_geometry()
    # print(coords_geom)

    # Setup bridges data
    # root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    # vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    # file_path = f"{root_source}/{vuln_bridges_rel_path}"
    # bridgesHandler = VulnerableBridgesHandler(spark, file_path)
    # bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()
    # print(bridges_coordinates_geometry)
>>>>>>> Stashed changes
