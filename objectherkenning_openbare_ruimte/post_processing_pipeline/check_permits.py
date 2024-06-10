from databricks.sdk.runtime import *
import pandas as pd
from difflib import get_close_matches
import requests
import re
from typing import List
from reference_db_connector import ReferenceDatabaseConnector

class DecosDataConnector(ReferenceDatabaseConnector):


    def is_container_permit(self, objects):
        """
        Check whether permit is for a container based on the 'objecten' column.
        """
        container_words = [
            "puinbak",
            "container",
            "keet",
            "cabin",
        ]

        try:
            for obj in objects:
                for word in container_words:
                    if any(get_close_matches(word, obj['object'].split())):
                        return True
        except Exception as e:
            print(f"There was an exception in the is_container_permit function: {e}")

        return False

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

            df = self.create_dataframe(rows, colnames)
            addresses = df["locatie"].tolist()  # Get list of addresses
            df = self.add_coordinates_columns(df, addresses)  # Add new columns for coordinates
            self.display_dataframe(df)
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")

    def process_rows_with_unsupported_data_type(self, rows, colnames):
        """
        Process the rows fetched from the database.
        """
        data = [dict(zip(colnames, row)) for row in rows]
        
        for record in data:
            for time_col in ['tijd_tvm_parkeervakken_van', 'tijd_tvm_parkeervakken_tot', 'tijd_tvm_stremmen_van', 'tijd_tvm_stremmen_tot']:
                if time_col in record:
                    record[time_col] = str(record[time_col])
        
        return data

    def create_dataframe(self, rows, colnames):
        """
        Create a DataFrame .
        """
        data = self.process_rows_with_unsupported_data_type(rows, colnames)
        df = pd.DataFrame(data, columns=colnames)

        # only keep rows with container data
        df['objecten'] = df['objecten'].apply(lambda x: json.loads(x) if x else [])
        df = df[df['objecten'].apply(self.is_container_permit)]
        return df

    def display_dataframe(self, df):
        """
        Display the DataFrame.
        """
        df.display()

    def convert_address_to_coordinates(self, address):
        """
        Convert address to coordinates using BAG API.
        """
        try:
            bag_url = "https://api.data.amsterdam.nl/atlas/search/adres/?q="

            street_and_number = ' '.join(address.split()[:2])
            with requests.get(bag_url + street_and_number) as response:
                bag_coords_lon_and_lat = json.loads(response.content)["results"][0]["centroid"]
                return [bag_coords_lon_and_lat[0], bag_coords_lon_and_lat[1]]
        except Exception as e:
            print(f"Error converting address into coordinates for {address}: {e}")
            return None


    def add_coordinates_columns(self, df, addresses):
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
            else:
                latitudes.append(None)
                longitudes.append(None)
        df["permit_lat"] = latitudes
        df["permit_lon"] = longitudes
        return df



az_tenant_id = "72fca1b1-2c2e-4376-a445-294d80196804"
db_scope = "keyvault"
db_host = "dev-bbn1-01-dbhost.postgres.database.azure.com"
db_name = "mdbdataservices"
db_port = "5432"
query = "SELECT id,kenmerk, locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '2024-02-17' AND datum_object_tm >= '2024-02-17'"

connector = DecosDataConnector(az_tenant_id, db_scope, db_host, db_name, db_port)
connector.run(query)

