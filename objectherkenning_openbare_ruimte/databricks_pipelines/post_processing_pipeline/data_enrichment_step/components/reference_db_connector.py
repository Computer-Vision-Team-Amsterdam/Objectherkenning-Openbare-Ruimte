import json
import subprocess
from abc import ABC, abstractmethod

import psycopg2
from databricks.sdk.runtime import *  # noqa: F403


class ReferenceDatabaseConnector(ABC):

    def __init__(self, az_tenant_id, db_host, db_name, db_port):
        self.az_tenant_id = az_tenant_id
        self.db_scope = "keyvault"
        self.db_host = db_host
        self.db_name = db_name
        self.db_port = db_port

        self.az_login_username = dbutils.secrets.get(  # noqa: F405
            scope=self.db_scope, key="app-reg-refdb-id"
        )
        self.az_login_password = dbutils.secrets.get(  # noqa: F405
            scope=self.db_scope, key="app-reg-refdb-key"
        )
        self.spn_refDb_username = "cvision_databricks"
        self.spn_refDb_password = None
        self._query_result_df = None  # set in run_query()

    def azure_login(self):
        command = [
            "az",
            "login",
            "--service-principal",
            "-u",
            self.az_login_username,
            "-p",
            self.az_login_password,
            "-t",
            self.az_tenant_id,
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
                print(f"{len(self.get_query_result_df())} object permits found.")

                conn.close()
        except Exception as e:
            print(f"Error: {e}")

    def get_query_result_df(self):
        return self.query_result_df
