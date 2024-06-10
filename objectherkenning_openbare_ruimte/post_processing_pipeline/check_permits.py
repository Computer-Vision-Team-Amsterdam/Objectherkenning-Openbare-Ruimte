import logging
import pandas as pd
from difflib import get_close_matches
import requests
import re
from typing import List
from abc import ABC, abstractmethod
import json
import subprocess
import psycopg2
from databricks_workspace import get_catalog_name
from reference_db_connector import ReferenceDatabaseConnector

# Suppress py4j INFO logs
logging.getLogger("py4j").setLevel(logging.WARNING)
 
class DecosDataConnector(ReferenceDatabaseConnector):

    def __init__(self):
       super().__init__() 
       self.catalog_name = get_catalog_name() # dpcv_dev or dpcv_prd

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

    def process_query_results(self, rows, colnames):
        """
        Process the results of the query.
        """
        df = self.create_dataframe(rows, colnames)
        
        df['objecten'] = df['objecten'].apply(lambda x: json.loads(x) if x else [])
        df = df[df['objecten'].apply(self.is_container_permit)] # Only keep rows with permits about containers
        df = df[df['locatie'].notnull()] # Only keep rows where location of the permit is not null
        addresses = df["locatie"].tolist()  # Get list of permit addresses in BAG format
        df = self.add_permit_coordinates_columns(df, addresses)  # Add new columns for lat lon coordinates of permits
        #self.write_null_coordinates_to_quarantine_table(df)
        self.write_healthy_df_to_table()
        self.display_dataframe(df)       
   
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
        return df     

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
                print(f"Warning: Unable to split Dutch street address using regex: {address}. Still trying to query the BAG API with the address...")
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
                latitudes.append(coordinates[0])
                longitudes.append(coordinates[1])
            else: # None, because there was an exception while converting the address into coordinates
                latitudes.append(None)
                longitudes.append(None)
        df["permit_lat"] = latitudes
        df["permit_lon"] = longitudes
        return df
    
    def write_null_coordinates_to_quarantine_table(self, df):
        """
        Write rows with null permit_lat or permit_lon to the 'quarantine' table in Databricks.
        
        Args:
        df (DataFrame): Input DataFrame containing the data.
        """
        # Filter rows with null permit_lat or permit_lon
        quarantine_df = df[df['permit_lat'].isnull() | df['permit_lon'].isnull()]

        # Write the filtered DataFrame to the 'quarantine' table in Databricks
        #quarantine_df.write.format("delta").mode("overwrite").saveAsTable("{env}.oor.{table_name_quarantine}")

    def write_healthy_df_to_table(df):
        """
        Write rows with non-null permit_lat and permit_lon to the 'good' table in Databricks.
        
        Args:
        df (DataFrame): Input DataFrame containing the data.
        """
        # Filter rows with non-null permit_lat and permit_lon
        healthy_df = df[df['permit_lat'].notnull() & df['permit_lon'].notnull()]

        # Write the filtered DataFrame to the 'good' table in Databricks
        #healthy_df.write.format("delta").mode("overwrite").saveAsTable("{env}.oor.{table_name}")

az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
db_scope = "keyvault"
db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
db_name = "mdbdataservices"
db_port = "5432"
query = "SELECT id,kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"

connector = DecosDataConnector(az_tenant_id, db_scope, db_host, db_name, db_port)
connector.run(query)
