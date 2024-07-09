import pandas as pd
from difflib import get_close_matches
import requests
import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException
import re
from typing import List, Tuple
import json
from shapely.geometry import Point
import numpy as np
import geopy.distance

from .databricks_workspace import get_catalog_name
from .reference_db_connector import ReferenceDatabaseConnector


class DecosDataHandler(ReferenceDatabaseConnector):

    def __init__(self, spark, az_tenant_id, db_host, db_name, db_port):
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark = spark
        self.catalog_name = get_catalog_name(self.spark)  
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
        print("Processing object permits (filter by object name, convert address to coordinates)...")
        self.query_result_df['objecten'] = self.query_result_df['objecten'].apply(lambda x: json.loads(x) if x else [])

        # Only keep rows with permits about containers
        self.query_result_df =  self.query_result_df[ self.query_result_df['objecten'].apply(self.is_container_permit)] 

        # Only keep rows where location of the permit is not null
        self.query_result_df = self.query_result_df[ self.query_result_df['locatie'].notnull()] 

         # Get list of permit addresses in BAG format
        addresses = self.query_result_df["locatie"].tolist() 

        # Add new columns for lat lon coordinates of permits
        self.query_result_df = self.add_permit_coordinates_columns(self.query_result_df, addresses)  

        display(self.query_result_df)

        # Store rows where permit_lat and permit_lon are non null as healthy data
        self._healthy_df = self.query_result_df[self.query_result_df['permit_lat'].notnull() & self.query_result_df['permit_lon'].notnull()]
        
        print(f"{len(self._healthy_df)} container permits with valid coordinates.")
        if len(self._healthy_df) == 0:
            raise ValueError("No permit coordinates could be converted from addresses.")

        # Store rows where permit_lat and permit_lon are null as quarantine data
        self._quarantine_df = self.query_result_df[self.query_result_df['permit_lat'].isnull() | self.query_result_df['permit_lon'].isnull()]
        print(f"{len(self._quarantine_df)} container permits with invalid coordinates.")

        

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
                print(f"Street and number: {street_and_number}")
            else:
                print(f"Warning: Unable to split Dutch street address using regex: {address}")
                street_and_number = address

            with requests.get(bag_url + street_and_number) as response:
                results = json.loads(response.content)["results"]
                print(f"Response content: {json.loads(response.content)}")
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
    
    def calculate_distances_to_closest_permits(self, permits_locations_as_points: List[Point], permits_ids: List[str], permits_coordinates: List[Tuple[float, float]], containers_locations_as_points: List[Point]):
        permit_distances = []
        closest_permits = []
        closest_permits_coordinates = []

        for container_location in containers_locations_as_points:
            closest_permit_distances = []
            for permit_location in permits_locations_as_points:
                try:
                    permit_dist = geopy.distance.distance(container_location.coords, permit_location.coords).meters
                except:
                    permit_dist = 0
                    print("Error occurred:")
                    print(f"Container location: {container_location}, {container_location.coords}")
                    print(f"Permit location: {permit_location}, {permit_location.coords}")
                closest_permit_distances.append(permit_dist)

            min_distance_idx = np.argmin(closest_permit_distances)
            permit_distances.append(float(closest_permit_distances[min_distance_idx]))
            closest_permits.append(permits_ids[min_distance_idx])
            closest_permits_coordinates.append(permits_coordinates[min_distance_idx])

        return permit_distances, closest_permits, closest_permits_coordinates
