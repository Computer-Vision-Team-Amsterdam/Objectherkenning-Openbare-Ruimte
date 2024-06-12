# Run clustering
# enrich with Decos data and with Bridges data 
# prioritize based on score
# store results in tables.

def get_databricks_environment():
    """
    Returns Productie, Ontwikkel or None based on the tags set in the subscription
    """

    tags = spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags")
    try:
        tags_json = json.loads(tags)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

    environment_tag = next((tag for tag in tags_json if tag.get("key") == "environment"), None)

    if environment_tag:
        environment = environment_tag.get("value")
        return environment   
    return None 

def get_catalog_name():
    """
    Sets the catalog name based on the environment retrieved from Databricks cluster tags
    """

    environment = get_databricks_environment()
    if environment is None:
        raise ValueError("Databricks environment is not set.")
    if environment == "Ontwikkel":
        catalog_name = "dpcv_dev"
    elif environment == "Productie":
        catalog_name = "dpcv_prd"
   
    return catalog_name  

def calculate_score(bridge_distance: float, permit_distance: float) -> float:
    """
    Calculate score for bridge and permit distance;
    """
    if permit_distance >= 40 and bridge_distance < 25:
        return 1 + max([(25 - bridge_distance) / 25, 0])
    elif permit_distance >= 40 and bridge_distance >= 25:
        return min(1.0, permit_distance / 100.0)
    else:
        return 0    

from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType

class Clustering:

    def __init__(self, date):
        
        self.catalog = get_catalog_name()
        self.schema = "oor"
        self.detection_metadata = spark.read.table(f'{self.catalog}.{self.schema}.bronze_detection_metadata') # TODO change table to silver_detection_metadata after implementing metadata healthcheck
        self.frame_metadata = spark.read.table(f'{self.catalog}.{self.schema}.bronze_frame_metadata') # TODO change table to silver_detection_metadata after implementing metadata healthcheck
        self._filter_objects_by_date(date)
        self._filter_objects_randomly()
        self.df_joined= self._join_frame_and_detection_metadata()

        self._containers_coordinates = self._extract_containers_coordinates()  
        self._containers_coordinates_geometry = self._convert_coordinates_to_point()

    def get_containers_coordinates(self):
        return self._containers_coordinates
    
    def get_containers_coordinates_geometry(self):
        return self._containers_coordinates_geometry

    def _filter_objects_by_date(self, date):
        self.detection_metadata = self.detection_metadata.filter(self.detection_metadata["image"].like(f"%{date}%"))

    def _filter_objects_randomly(self):
        self.detection_metadata = self.detection_metadata.sample(False, 0.1, seed=42)  # 10% sample size

    def _filter_objects_by_tracking_id(self):
        pass   

    def _join_frame_and_detection_metadata(self):

        joined_df = self.frame_metadata.join(self.detection_metadata, self.frame_metadata["image"] == self.detection_metadata["image"])
        filtered_df = joined_df.filter(self.frame_metadata["image"] == self.detection_metadata["image"])
        columns = ["gps_date", "id", "object_class", "gps_lat", "gps_lon"]
        selected_df = filtered_df.select(columns)
        joined_df = selected_df \
                    .withColumnRenamed("gps_date", "detection_date") \
                    .withColumnRenamed("id", "detection_id")

        return joined_df
    
    def _extract_containers_coordinates(self):

        # Collect the DataFrame rows as a list of Row objects
        rows = self.df_joined.select("gps_lat", "gps_lon").collect()
    
        # Convert the list of Row objects into a list of tuples
        containers_coordinates = [(row['gps_lat'], row['gps_lon']) for row in rows]

        return containers_coordinates
           
    def _convert_coordinates_to_point(self):
        """
        We need the containers coordinates as Point to perform distance calculations
        """
        containers_coordinates_geometry = [Point(location) for location in self._containers_coordinates] 
        return containers_coordinates_geometry

    # why is this operation so complicated!?    
    def add_column(self, column_name, values, data_type=StringType()):

        schema=StructType([StructField(column_name, data_type, True)])
        # Ensure the length of the values matches the number of rows in the dataframe
        if len(values) != self.df_joined.count():
            raise ValueError("The length of the list does not match the number of rows in the dataframe")

        #convert list to a dataframe    
        b = sqlContext.createDataFrame([(v,) for v in values], [column_name])

        #add 'sequential' index and join both dataframe to get the final result
        self.df_joined = self.df_joined.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
        b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

        self.df_joined = self.df_joined.join(b, self.df_joined.row_idx == b.row_idx).\
                    drop("row_idx")

from abc import ABC, abstractmethod
import json
import subprocess
import psycopg2

class ReferenceDatabaseConnector(ABC):

    def __init__(self, az_tenant_id, db_scope, db_host, db_name, db_port):
        self.az_tenant_id = az_tenant_id
        self.db_scope = db_scope
        self.db_host = db_host
        self.db_name = db_name
        self.db_port = db_port

        self.az_login_username = dbutils.secrets.get(scope=self.db_scope, key="app-reg-refdb-id")
        self.az_login_password = dbutils.secrets.get(scope=self.db_scope, key="app-reg-refdb-key")
        self.spn_refDb_username = dbutils.secrets.get(scope=self.db_scope, key="referenceDatabaseSpnUsername")
        self.spn_refDb_password = None
        self._query_result_df = None # set in run_query()

    def azure_login(self):
        command = [
            "az", "login", "--service-principal", "-u", self.az_login_username, 
            "-p", self.az_login_password, "-t", self.az_tenant_id
        ]
        subprocess.check_call(command)

    def retrieve_access_token(self):
        command = ["az", "account", "get-access-token", "--resource-type", "oss-rdbms"]
        output = subprocess.check_output(command)
        token_info = json.loads(output)
        self.spn_refDb_password = token_info["accessToken"]

    def connect_to_database(self):
        conn_string = (
            f"host='{self.db_host}' dbname='{self.db_name}' "
            f"user='{self.spn_refDb_username}' password='{self.spn_refDb_password}'"
        )
        try:
            conn = psycopg2.connect(conn_string)
            print("Connection to the database was successful.")
            return conn
        except psycopg2.Error as e:
            print(f"Database connection error: {e}")
            return None
        
    @abstractmethod
    def run_query(self, conn, query):
        pass


    @abstractmethod
    def create_dataframe(self, rows, colnames):
        """
        Create dataframe from query result.
        """
        pass

    def run_query(self, conn, query):
        """
        Run the SQL query and process the results.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            cursor.close()
            self._query_result_df = self.create_dataframe(rows, colnames)
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")

    def run(self, query):
        """
        Execute the full workflow: Azure login, retrieve token, connect to DB, run query.
        """
        try:
            self.azure_login()
            self.retrieve_access_token()
            conn = self.connect_to_database()
            if conn:
                self.run_query(conn, query)
                conn.close()
        except Exception as e:
            print(f"Error: {e}")

    def get_query_result_df(self):
        return self.query_result_df

import pandas as pd
from difflib import get_close_matches
import requests
import re
from typing import List
from abc import ABC, abstractmethod
import json
import subprocess
import psycopg2 
from shapely.geometry import LineString, Point
import numpy as np

class DecosDataHandler(ReferenceDatabaseConnector):

    def __init__(self, az_tenant_id, db_scope, db_host, db_name, db_port):
        super().__init__(az_tenant_id, db_scope, db_host, db_name, db_port)
        self.catalog_name = get_catalog_name()  

        self.query_result_df = None

    def is_container_permit(self, objects):
        """
        Check whether permit is for a container based on the 'objecten' column.
        """
        container_words = ["puinbak", "container", "keet", "cabin",]

        try:
            for obj in objects:
                for word in container_words:
                    if any(get_close_matches(word, obj['object'].split())):
                        return True
        except Exception as e:
            print(f"There was an exception in the is_container_permit function: {e}")

        return False

    def process_query_result(self):
        """
        Process the results of the query.
        """
        
        self.query_result_df['objecten'] = self.query_result_df['objecten'].apply(lambda x: json.loads(x) if x else [])

        # Only keep rows with permits about containers
        self.query_result_df =  self.query_result_df[ self.query_result_df['objecten'].apply(self.is_container_permit)] 

        # Only keep rows where location of the permit is not null
        self.query_result_df = self.query_result_df[ self.query_result_df['locatie'].notnull()] 

         # Get list of permit addresses in BAG format
        addresses = self.query_result_df["locatie"].tolist() 

        # Add new columns for lat lon coordinates of permits
        self.query_result_df = self.add_permit_coordinates_columns(self.query_result_df, addresses)  

        # Store rows where permit_lat and permit_lon are non null as healthy data
        self._healthy_df = self.query_result_df[self.query_result_df['permit_lat'].notnull() & self.query_result_df['permit_lon'].notnull()]

        # Store rows where permit_lat and permit_lon are null as quarantine data
        self._quarantine_df = self.query_result_df[self.query_result_df['permit_lat'].isnull() | self.query_result_df['permit_lon'].isnull()]

        self._permits_coordinates = self._extract_permits_coordinates()  
        self._permits_coordinates_geometry = self._convert_coordinates_to_point()

        #self.write_healthy_df_to_table()
        #self.write_qurantine_df_to_table()

    # TODO create the dataframe with spark!
    def create_dataframe(self, rows, colnames):
        """
        Create a DataFrame .
        """

        def _load_columns_with_unsupported_data_type():
            """
            Pandas is complaining about loading the types of the following columns so we load them as strings.
            """
            data = [dict(zip(colnames, row)) for row in rows]
            
            for record in data:
                for time_col in ['tijd_tvm_parkeervakken_van', 'tijd_tvm_parkeervakken_tot', 'tijd_tvm_stremmen_van', 'tijd_tvm_stremmen_tot']:
                    if time_col in record:
                        record[time_col] = str(record[time_col])
            
            return data    
            
        data = _load_columns_with_unsupported_data_type()
        df = pd.DataFrame(data, columns=colnames)
        
        self.query_result_df = df     

    def display_dataframe(self, df):
        """
        Display the DataFrame.
        """
        df.display()

    def split_dutch_street_address(self, raw_address: str) -> List[str]:
        """
        This function separates an address string (street name, house number, house number extension and zipcode)
        into parts.
        Regular expression quantifiers:
        X?  X, once or not at all
        X*  X, zero or more times
        X+  X, one or more times
        """

        regex = "(\D+)\s+(\d+)\s?(.*)\s+(\d{4}\s*?[A-z]{2})"
        return re.findall(regex, raw_address)
    
    def convert_address_to_coordinates(self, address):
        """
        Convert address to coordinates using BAG API.
        """
        response = None
 
        try:
            bag_url = "https://api.data.amsterdam.nl/atlas/search/adres/?q="
            split_dutch_address = self.split_dutch_street_address(address)
            if split_dutch_address:
                street_and_number = split_dutch_address[0][0] + " " + split_dutch_address[0][1]
            else:
                print(f"Warning: Unable to split Dutch street address using regex: {address}")
                street_and_number = address

            with requests.get(bag_url + street_and_number) as response:
                results = json.loads(response.content)["results"]
                for result in results:
                    if "centroid" in result:
                        bag_coords_lon_and_lat = result["centroid"]
                        return [bag_coords_lon_and_lat[0], bag_coords_lon_and_lat[1]]
                # If no centroid is found in any result
                return None    
        except Exception as e:
            print(f"Error converting address into coordinates for {address}: {e}")
            print(f"Street and number: {split_dutch_address}")
            print(f'Response from server is {json.loads(response.content)["results"]}')
      
            return None

    def add_permit_coordinates_columns(self, df, addresses):
        """
        Add new columns "permit_lat" and "permit_lon" to the DataFrame and populate them with latitude and longitude values fetched from the BAG API.
        """
        latitudes = []
        longitudes = []
        for address in addresses:
            coordinates = self.convert_address_to_coordinates(address)
            if coordinates:
                longitudes.append(coordinates[0])
                latitudes.append(coordinates[1])
            else: # None, because there was an exception while converting the address into coordinates
                longitudes.append(None)
                latitudes.append(None)
                
        df["permit_lat"] = latitudes
        df["permit_lon"] = longitudes
        return df
    
    def write_quarantine_df_to_table(self, df):
        """
        Write rows with null permit_lat or permit_lon to the 'quarantine' table in Databricks.
        
        Args:
        df (DataFrame): Input DataFrame containing the data.
        """
        # Write the filtered DataFrame to the 'quarantine' table in Databricks
        #quarantine_df.write.format("delta").mode("overwrite").saveAsTable("{env}.oor.{table_name_quarantine}")

    def write_healthy_df_to_table(self, healthy_df):
        """
        Write rows with non-null permit_lat and permit_lon to the 'good' table in Databricks.
        
        Args:
        df (DataFrame): Input DataFrame containing the data.
        """
        # Write the filtered DataFrame to the 'good' table in Databricks
        #healthy_df.write.format("delta").mode("overwrite").saveAsTable("{env}.oor.{table_name}")

    def _extract_permits_coordinates(self):
        # -----> for spark dataframes
        # # Collect the DataFrame rows as a list of Row objects
        # rows = self._healthy_df.select("permit_lat", "permit_lon").collect()
    
        # # Convert the list of Row objects into a list of tuples
        # permits_coordinates = [(row['permit_lat'], row['permit_lon']) for row in rows]

        # ------> for pandas dataframes

        permits_coordinates = list(self._healthy_df[['permit_lat', 'permit_lon']].itertuples(index=False, name=None))

        return permits_coordinates
    
    def get_permits_ids(self):
        # -----> for spark dataframes
        # # Collect the DataFrame rows as a list of Row objects
        # rows = self._healthy_df.select("id").collect()

        # # Convert the list of Row objects into a list of tuples
        # permits_ids = [(row['id']) for row in rows]

        # ------> for pandas dataframes
        permits_ids = self._healthy_df['id'].tolist()

        return permits_ids
           
    def _convert_coordinates_to_point(self):
        """
        We need the permits coordinates as Point to perform distance calculations
        """
        permits_coordinates_geometry = [Point(location) for location in self._permits_coordinates] 
        return permits_coordinates_geometry    

    def get_permits_coordinates(self):
        return self._permits_coordinates
    
    def get_permits_coordinates_geometry(self):
        return self._permits_coordinates_geometry
    
    def get_healthy_df(self):
        return self._healthy_df

    def get_quarantine_df(self):
        return self._quarantine_df
    
    def calculate_distances_to_closest_permits(permits_locations_as_points: List[Point], permits_ids: str, containers_locations_as_points: List[Point]):
        permit_distances = []
        closest_permits = [] # TODO rename this!
        for container_location in containers_locations_as_points:
            closest_permit_distances = []
            for permit_location in permits_locations_as_points:
                try:
                    permit_dist = geopy.distance.distance(container_location.coords, permit_location.coords).meters
                except:
                    permit_dist = 0
                    print("Error occured:")
                    print(f"Container location: {container_location}, {container_location.coords}")
                    print(f"Permit location: {permit_location}, {permit_location.coords}")
                closest_permit_distances.append(permit_dist)
            permit_distances.append(np.amin(closest_permit_distances))
            closest_permits.append(permits_ids[np.argmin(closest_permit_distances)])

            # otherwise spark complains about float64 not being supported
            permit_distances = [float(p) for p in permit_distances]

        return permit_distances, closest_permits

    
import os
import geojson
from tqdm import tqdm
from typing import List, Tuple, Union
from osgeo import osr  # pylint: disable-all
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import geopy.distance

class VulnerableBridgesHandler:
    def __init__(self, file_path):
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
        sparkDataFrame = spark.read.format("json") \
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



if __name__ == "__main__":

    ########## SETUP ##########    
    # Setup clustering
    clustering = Clustering(date="D14M03Y2024")  
    containers_coordinates_geometry = clustering.get_containers_coordinates_geometry()
    
    # Setup bridges data
    root_source = f"abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net"
    vuln_bridges_rel_path = "test-diana/vuln_bridges.geojson"
    file_path = f"{root_source}/{vuln_bridges_rel_path}"
    bridgesHandler = VulnerableBridgesHandler(file_path)
    bridges_coordinates_geometry = bridgesHandler.get_bridges_coordinates_geometry()

    # Setup permit data
    az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
    db_scope = "keyvault"
    db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
    db_name = "mdbdataservices"
    db_port = "5432"
 
    decosDataHandler = DecosDataHandler(az_tenant_id, db_scope, db_host, db_name, db_port)

    ######### ENRICHMENTS ###########

    # Enrich with bridges data
    closest_bridges_distances = VulnerableBridgesHandler.calculate_distances_to_closest_vulnerable_bridges(
        bridges_locations_as_linestrings=bridgesHandler.get_bridges_coordinates_geometry(),
        containers_locations_as_points=clustering.get_containers_coordinates_geometry())
    
    clustering.add_column(column_name="closest_bridge_distance", values=closest_bridges_distances)

    # Enrich with decos data
    query = "SELECT id, kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"
    decosDataHandler.run(query)
    query_result_df = decosDataHandler.get_query_result_df()
    decosDataHandler.process_query_result()
    permit_distances, closest_permits = DecosDataHandler.calculate_distances_to_closest_permits(
        permits_locations_as_points=decosDataHandler.get_permits_coordinates_geometry(),
        permits_ids=decosDataHandler.get_permits_ids(),
        containers_locations_as_points=containers_coordinates_geometry)

    clustering.add_column(column_name="closest_permit_distance", values=permit_distances, data_type=FloatType())
    clustering.add_column(column_name="closest_permit_id", values=closest_permits)

    # Enrich with score 

    scores = [
            float(calculate_score(closest_bridges_distances[idx], permit_distances[idx]))
            for idx in range(len(clustering.get_containers_coordinates()))
        ]
    clustering.add_column(column_name="score", values=scores)

    display(clustering.df_joined)



    

