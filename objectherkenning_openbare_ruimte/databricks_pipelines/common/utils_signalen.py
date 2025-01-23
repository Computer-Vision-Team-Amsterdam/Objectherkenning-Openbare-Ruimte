import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from databricks.sdk.runtime import *  # noqa: F403
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.aggregators.silver_metadata_aggregator import (
    SilverMetadataAggregator,
)


class SignalConnectionConfigurer:
    """
    Manages connection details for signal operations based on environment.

    This class sets connection details such as token URL, base URL, and client secret
    based on the environment tag extracted from Spark configuration.

    Client credentials are used to authenticate with Keycloak.
    A POST request is sent to the Keycloak token endpoint to receive an access token.
    This token is then used to authenticate subsequent API
    requests by including it in the Authorization header.

    # Use these details to authenticate the requests to the correct API endpoint.
        signalHandler = SignalHandler(base_url, access_token)

    """

    def __init__(
        self,
        spark: SparkSession,
        client_id,
        client_secret_name,
        access_token_url,
        base_url,
    ):
        self.spark = spark
        self._client_id = client_id
        self._client_secret_name = client_secret_name
        self.access_token_url = access_token_url
        self.base_url = base_url

    def get_base_url(self):
        return self.base_url

    def get_client_id(self):
        return self._client_id

    def _get_client_secret(self):
        return dbutils.secrets.get(  # noqa: F405
            scope="keyvault", key=self._client_secret_name
        )  # noqa: F405

    def get_access_token(self) -> Any:
        payload = {
            "client_id": self.get_client_id(),
            "client_secret": self._get_client_secret(),
            "grant_type": "client_credentials",
        }
        response = requests.post(self.access_token_url, data=payload, timeout=60)
        if response.status_code == 200:
            print("The server successfully answered the access token request.")
            return response.json()["access_token"]
        else:
            return response.raise_for_status()


class SignalHandler:

    def __init__(
        self,
        spark,
        catalog,
        schema,
        device_id,
        client_id,
        client_secret_name,
        access_token_url,
        base_url,
    ):

        self.spark = spark
        self.device_id = device_id
        self.catalog_name = catalog
        self.schema = schema

        self.api_max_upload_size = 20 * 1024 * 1024  # 20MB = 20*1024*1024
        signalConnectionConfigurer = SignalConnectionConfigurer(
            spark, client_id, client_secret_name, access_token_url, base_url
        )
        self.base_url: str = signalConnectionConfigurer.get_base_url()  # type: ignore
        access_token = signalConnectionConfigurer.get_access_token()
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {access_token}"}  # type: ignore
        self.verify_ssl = True

    def get_signal(self, sig_id: str) -> Any:
        """
        Retrieves the details of an existing signal.

        This method sends a GET request to retrieve the details of the signal identified by `sig_id`.

        Parameters
        ----------
        sig_id : str
            The ID of the signal to be retrieved.

        Returns
        -------
        Any
            The response JSON containing the details of the signal if the request is successful.

        Raises
        ------
        HTTPError
            If the server responds with a status code other than 200 (OK),
            an HTTPError will be raised with the response status and message.
        """
        response = requests.get(
            self.base_url + f"/{sig_id}",
            headers=self.headers,
            verify=self.verify_ssl,
            timeout=60,
        )

        if response.status_code == 200:
            print("The server successfully performed the GET request.")
            return response.json()
        return response.raise_for_status()

    def post_signal(self, json_content: Any) -> Any:
        """
        Sends a POST request to create a new signal with the provided JSON content.

        Parameters
        ----------
        json_content : Any
            The JSON content to be sent in the body of the POST request.

        Returns
        -------
        Any
            The ID of the created signal if the request is successful.

        Raises
        ------
        HTTPError
            If the server responds with a status code other than 201 (Created),
            an HTTPError will be raised with the response status and message.
        """
        response = requests.post(
            self.base_url,
            json=json_content,
            headers=self.headers,
            verify=self.verify_ssl,
            timeout=60,
        )

        if response.status_code == 201:
            print(
                "The server successfully performed the POST request and created an incident."
            )
            return response.json()["id"]
        else:
            return response.raise_for_status()

    # Patch requests do not work at the moment. Throws 500 Internal Error.
    def patch_signal(self, sig_id: str, text_note: str) -> Any:
        """
        Updates an existing signal by adding notes.

        This method sends a PATCH request to update the signal identified by `sig_id`
        with the provided `text_note`. Adding notes during signal creation is not supported,
        hence this method is used to add instructions for ASC and BOA on the street.

        Parameters
        ----------
        sig_id : str
            The ID of the signal to be updated.
        text_note : str
            The text note to be added to the signal.

        Returns
        -------
        Any
            The response JSON from the server if the request is successful.

        Raises
        ------
        HTTPError
            If the server responds with a status code other than 200 (OK),
            an HTTPError will be raised with the response status and message.
        """
        json_content = {"notes": [{"text": text_note}]}

        response = requests.patch(
            self.base_url + f"/{sig_id}",
            json=json_content,
            headers=self.headers,
            verify=self.verify_ssl,
            timeout=60,
        )

        if response.status_code == 200:
            print("The server successfully performed the PATCH request.")
            return response.json()
        else:
            return response.raise_for_status()

    def image_upload(self, filename: str, sig_id: str) -> Any:
        """
        Uploads an image file as an attachment to an existing signal.

        This method sends a POST request to upload an image file to the signal identified by `sig_id`.
        The file size is checked against the maximum allowed upload size.

        Parameters
        ----------
        filename : str
            The path to the image file to be uploaded.
        sig_id : str
            The ID of the signal to which the image file will be attached.

        Returns
        -------
        Any
            The response JSON from the server if the upload is successful.

        Raises
        ------
        Exception
            If the file size exceeds the maximum allowed upload size.
        HTTPError
            If the server responds with a status code other than 201 (Created),
            an HTTPError will be raised with the response status and message.
        """
        if os.path.getsize(filename) > self.api_max_upload_size:
            msg = f"File can be a maximum of {self.api_max_upload_size} bytes in size."
            raise Exception(msg)

        files = {"file": (filename, open(filename, "rb"))}

        response = requests.post(
            self.base_url + f"/{sig_id}/attachments/",
            files=files,
            headers=self.headers,
            verify=self.verify_ssl,
            timeout=60,
        )

        if response.status_code == 201:
            print(
                "The server successfully performed the POST request and uploaded an image."
            )
            return response.json()
        else:
            return response.raise_for_status()

    def post_signal_with_image_attachment(self, json_content: Any, filename: str):
        signal_id = self.post_signal(json_content=json_content)
        self.image_upload(filename=filename, sig_id=signal_id)
        return signal_id

    @staticmethod
    def get_signal_description() -> str:
        """
        Text we send when creating a notification.
        """
        return (
            "=== TEST SIGNAL CVT, please IGNORE === \n\n"
            "Dit is een automatisch gegenereerd signaal: Met behulp van beeldherkenning is een bouwcontainer of "
            "bouwkeet gedetecteerd op onderstaande locatie, waar waarschijnlijk geen vergunning voor is. N.B. Het "
            "adres betreft een schatting van het dichtstbijzijnde adres bij de containerlocatie, er is geen "
            "informatie bekend in hoeverre dit het adres is van de containereigenaar."
        )

    @staticmethod
    def get_text_general_instruction_note(
        permit_distance: str, bridge_distance: str
    ) -> str:
        """
        Text we send when editing a notification by adding a note.
        """
        return (
            f"Categorie Rood: 'mogelijk illegaal object op kwetsbare kade'\n"
            f"Afstand tot kwetsbare kade: {bridge_distance} meter\n"
            f"Afstand tot objectvergunning: {permit_distance} meter"
        )

    @staticmethod
    def get_text_asc_instruction_note() -> str:
        """
        Text we send when editing a notification by adding a note.
        """
        return (
            "Instructie ASC:\n"
            "(i) Foto bekijken en alleen signalen doorzetten naar THOR indien er inderdaad een "
            "bouwcontainer of bouwkeet op de foto staat. \n "
            "(ii) De bron voor dit signaal moet op 'Automatische signalering' blijven staan, zodat BOA's dit "
            "signaal herkennen in City Control onder 'Signalering'."
        )

    @staticmethod
    def get_text_boa_instruction_note() -> str:
        """
        Text we send when editing a notification by adding a note.
        """
        return (
            "Instructie BOA’s:\n "
            "(i) Foto bekijken en beoordelen of dit een bouwcontainer of bouwkeet is waar vergunningsonderzoek "
            "ter plaatse nodig is.\n"
            "(ii) Check Decos op aanwezige vergunning voor deze locatie of vraag de vergunning op bij "
            "containereigenaar.\n "
            "(iii) Indien geen geldige vergunning, volg dan het reguliere handhavingsproces."
        )

    @staticmethod
    def get_bag_address_in_range(
        longitude: float, latitude: float, max_building_search_radius=50
    ) -> List[Optional[str]]:
        """
        Retrieves the nearest building information in BAG for a given location point within a specified search radius.

        Parameters
        ----------
        lat : float
            The latitude coordinate of the location point.
        lon : float
            The longitude coordinate of the location point.
        max_building_search_radius : int, optional
            The maximum radius (in meters) to search for buildings around the given location point.
            Default is 50 meters.

        Returns
        -------
        List[Optional[str]]
            A list containing the address details of the nearest building within the specified search radius.
            The list contains the following elements:
            - Street name (openbare_ruimte)
            - House number (huisnummer)
            - Postcode

        Example
        --------
        lat = 52.3702
        lon = 4.8952
        _get_bag_address_in_range(lat, lon)
        ['Dam Square', '1', '1012 JS']
        """
        location_point = {"lat": latitude, "lon": longitude}
        bag_url = (
            f"https://api.data.amsterdam.nl/bag/v1.1/nummeraanduiding/"
            f"?format=json&locatie={location_point['lat']},{location_point['lon']},"
            f"{max_building_search_radius}&srid=4326&detailed=1"
        )

        response = requests.get(bag_url, timeout=60)
        if response.status_code == 200:
            response_content = json.loads(response.content)
            if response_content["count"] > 0:
                # Get first element
                first_element = json.loads(response.content)["results"][0]
                return [
                    first_element["openbare_ruimte"]["_display"],
                    first_element["huisnummer"],
                    first_element["postcode"],
                ]
            else:
                print(
                    f"No BAG address in the range of {max_building_search_radius} found."
                )
                return []
        else:
            print(
                f"Failed to get address from BAG, status code {response.status_code}."
            )
            return []

    @staticmethod
    def fill_incident_details(incident_date: str, lon: float, lat: float) -> Any:
        """
        Fills in the details of an incident and returns a JSON object representing the incident.

        Parameters
        ----------
        incident_date : str
            The date of the incident in the format 'YYYY-MM-DD'.
        lon : float
            Longitude coordinate of the incident location.
        lat : float
            Latitude coordinate of the incident location.

        Returns
        -------
        Any
            A JSON object representing the incident details including text description,
            location coordinates, category, reporter email, priority, and incident start date.

        Example:
        --------
        incident_date = "2024-05-30"
        lon = 4.8952
        lat = 52.3702
        json_to_send = _fill_incident_details(incident_date, lat_lon)
        """

        date_now = datetime.strptime(incident_date, "%Y-%m-%d").date()
        bag_address = SignalHandler.get_bag_address_in_range(
            longitude=lon, latitude=lat
        )

        json_to_send = {
            "text": SignalHandler.get_signal_description(),
            "location": {
                "geometrie": {
                    "type": "Point",
                    "coordinates": [lon, lat],  # LON-LAT in this order!
                },
            },
            "category": {
                "sub_category": "/signals/v1/public/terms/categories/overlast-in-de-openbare-ruimte/"
                "sub_categories/hinderlijk-geplaatst-object"
            },
            "reporter": {"email": "cvt@amsterdam.nl"},
            "priority": {
                "priority": "normal",
            },
            "source": "Automatische signalering",
            "incident_date_start": date_now.strftime("%Y-%m-%d %H:%M"),
        }
        if bag_address:
            location_json = {
                "location": {
                    "geometrie": {
                        "type": "Point",
                        "coordinates": [lon, lat],  # LON-LAT in this order!
                    },
                    "address": {
                        "openbare_ruimte": bag_address[0],
                        "huisnummer": bag_address[1],
                        "woonplaats": "Amsterdam",
                    },
                }
            }
            # Only add 'postcode' if bag_address[2] is not None
            if bag_address[2] is not None:
                location_json["location"]["address"]["postcode"] = bag_address[2]  # type: ignore
            json_to_send.update(location_json)

        return json_to_send

    def process_notifications(self, top_scores_df):
        date_of_notification = datetime.today().strftime("%Y-%m-%d")
        top_scores_df_with_date = top_scores_df.withColumn(
            "notification_date", F.to_date(F.lit(date_of_notification))
        )
        silverFrameAndDetectionMetadata = SilverMetadataAggregator(
            spark=self.spark, catalog=self.catalog_name, schema=self.schema
        )
        successful_notifications = []
        unsuccessful_notifications = []

        for entry in top_scores_df_with_date.collect():
            LAT = float(entry["object_lat"])
            LON = float(entry["object_lon"])
            detection_id = entry["detection_id"]
            image_upload_path = (
                silverFrameAndDetectionMetadata.get_image_upload_path_from_detection_id(
                    detection_id=detection_id,
                    device_id=self.device_id,
                )
            )
            entry_dict = entry.asDict()
            entry_dict.pop("processed_at", None)
            entry_dict.pop("id", None)

            try:
                dbutils.fs.head(image_upload_path)  # noqa: F405
                notification_json = self.fill_incident_details(
                    incident_date=date_of_notification, lon=LON, lat=LAT
                )
                signal_id = self.post_signal_with_image_attachment(
                    json_content=notification_json, filename=image_upload_path
                )
                print(
                    f"Created signal {signal_id} with image on {date_of_notification} with lat {LAT} and lon {LON}.\n\n"
                )
                entry_dict["signal_id"] = signal_id
                updated_entry = Row(**entry_dict)
                successful_notifications.append(updated_entry)
            except Exception as e:
                entry_dict.pop("notification_date", None)
                updated_failed_entry = Row(**entry_dict)
                if "java.io.FileNotFoundException" in str(e):
                    print(
                        f"Image not found: {image_upload_path}. Skip creating notification...\n\n"
                    )
                else:
                    print(f"An error occurred: {e}\n\n")
                unsuccessful_notifications.append(updated_failed_entry)

        return successful_notifications, unsuccessful_notifications
