import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import requests
from databricks.sdk.runtime import *  # noqa: F403
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.aggregators.silver_metadata_aggregator import (
    SilverMetadataAggregator,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (
    SilverDetectionMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils_images import (
    OutputImage,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step import (
    BENKAGGConnector,
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
        spark_session: SparkSession,
        client_id: str,
        client_secret_name: str,
        access_token_url: str,
        base_url: str,
    ):
        self.spark_session = spark_session
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
        spark_session: SparkSession,
        catalog: str,
        schema: str,
        device_id: str,
        signalen_settings: Dict[str, str],
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        object_classes: Dict[int, str],
        annotate_images: bool = False,
    ):
        self.spark_session = spark_session
        self.device_id = device_id
        self.catalog_name = catalog
        self.schema = schema
        self.object_classes = object_classes
        self.annotate_images = annotate_images
        self.api_max_upload_size = 20 * 1024 * 1024  # 20MB = 20*1024*1024

        client_id = signalen_settings["client_id"]
        client_secret_name = signalen_settings["client_secret_name"]
        access_token_url = signalen_settings["access_token_url"]
        base_url = signalen_settings["base_url"]
        signalConnectionConfigurer = SignalConnectionConfigurer(
            spark_session, client_id, client_secret_name, access_token_url, base_url
        )
        self.base_url: str = signalConnectionConfigurer.get_base_url()  # type: ignore
        access_token = signalConnectionConfigurer.get_access_token()
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {access_token}"}  # type: ignore
        self.verify_ssl = True

        self.bankAggConnector = BENKAGGConnector(
            az_tenant_id=az_tenant_id,
            db_host=db_host,
            db_name=db_name,
        )

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

    def image_upload(self, image: OutputImage, sig_id: str) -> Any:
        """
        Uploads an image file as an attachment to an existing signal.

        This method sends a POST request to upload an image file to the signal identified by `sig_id`.
        The file size is checked against the maximum allowed upload size.

        Parameters
        ----------
        image : OutputImage
            The image to be uploaded.
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
        filename = image.get_image_path()

        if os.path.getsize(filename) > self.api_max_upload_size:
            msg = f"File can be a maximum of {self.api_max_upload_size} bytes in size."
            raise Exception(msg)

        files = {"file": (filename, image.to_bytes())}

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

    def post_signal_with_image_attachment(self, json_content: Any, image: OutputImage):
        signal_id = self.post_signal(json_content=json_content)
        self.image_upload(image=image, sig_id=signal_id)
        return signal_id

    @staticmethod
    def get_signal_description(object_class_str: str) -> str:
        """
        Text we send when creating a notification.
        """
        return (
            f"Dit is een automatisch gegenereerd signaal: Met behulp van beeldherkenning is een {object_class_str} "
            "gedetecteerd op onderstaande locatie, waar waarschijnlijk geen vergunning voor is. N.B. Het "
            "adres betreft een schatting van het dichtstbijzijnde adres bij de objectlocatie, er is geen "
            "informatie bekend in hoeverre dit het adres is van de objecteigenaar."
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
    def get_text_asc_instruction_note(object_class_str: str) -> str:
        """
        Text we send when editing a notification by adding a note.
        """
        return (
            "Instructie ASC:\n"
            "(i) Foto bekijken en alleen signalen doorzetten naar THOR indien er inderdaad een "
            f"{object_class_str} op de foto staat. \n "
            "(ii) De bron voor dit signaal moet op 'Automatische signalering' blijven staan, zodat BOA's dit "
            "signaal herkennen in City Control onder 'Signalering'."
        )

    @staticmethod
    def get_text_boa_instruction_note(object_class_str: str) -> str:
        """
        Text we send when editing a notification by adding a note.
        """
        return (
            "Instructie BOA’s:\n "
            f"(i) Foto bekijken en beoordelen of dit een {object_class_str} is waar vergunningsonderzoek "
            "ter plaatse nodig is.\n"
            "(ii) Check Decos op aanwezige vergunning voor deze locatie of vraag de vergunning op bij "
            "objecteigenaar.\n "
            "(iii) Indien geen geldige vergunning, volg dan het reguliere handhavingsproces."
        )

    def get_bag_address_in_range(
        self, longitude: float, latitude: float, max_building_search_radius=50
    ) -> Union[List[Union[str, int]], None]:
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
        get_bag_address_in_range(lat, lon)
        ['Dam Square', '1', '1012 JS']
        """
        bag_url = (
            f"https://api.data.amsterdam.nl/geosearch/?datasets=benkagg/adresseerbareobjecten"
            f"&lat={latitude}&lon={longitude}&radius={max_building_search_radius}"
        )

        response = requests.get(bag_url, timeout=60)
        if response.status_code == 200:
            response_content = json.loads(response.content)
            if len(response_content["features"]) > 0:
                first_element_id = response_content["features"][0]["properties"]["id"]
                result_df = (
                    self.bankAggConnector.get_benkagg_adresseerbareobjecten_by_id(
                        first_element_id
                    )
                )
                if result_df.empty:
                    print(f"Warning: No results found for id: {first_element_id}")
                    return None
                if result_df.shape[0] > 1:
                    print(f"Warning: Multiple results found for id: {first_element_id}")
                return [
                    result_df["openbareruimte_naam"].iloc[0],
                    int(result_df["huisnummer"].iloc[0]),
                    result_df["postcode"].iloc[0],
                ]
            else:
                print(
                    f"No BAG address in the range of {max_building_search_radius} found."
                )
        else:
            print(
                f"Failed to get address from BAG, status code {response.status_code}."
            )
        return None

    def fill_incident_details(
        self, incident_date: str, lon: float, lat: float, object_class_str: str
    ) -> Any:
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
        json_to_send = {
            "text": SignalHandler.get_signal_description(object_class_str),
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

        bag_address = self.get_bag_address_in_range(longitude=lon, latitude=lat)
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

    def process_notifications(
        self,
        top_scores_df: DataFrame,
    ):
        date_of_notification = datetime.today().strftime("%Y-%m-%d")
        top_scores_df_with_date = top_scores_df.withColumn(
            "notification_date", F.to_date(F.lit(date_of_notification))
        )
        silverFrameAndDetectionMetadata = SilverMetadataAggregator(
            spark_session=self.spark_session,
            catalog=self.catalog_name,
            schema=self.schema,
        )
        successful_notifications = []
        unsuccessful_notifications = []

        for entry in top_scores_df_with_date.collect():
            LAT = float(entry["object_lat"])
            LON = float(entry["object_lon"])
            detection_id = entry["detection_id"]
            object_class = entry["object_class"]
            object_class_str = self.object_classes.get(object_class)
            image_upload_path = (
                silverFrameAndDetectionMetadata.get_image_upload_path_from_detection_id(
                    detection_id=detection_id,
                    device_id=self.device_id,
                )
            )

            entry_dict = entry.asDict()
            entry_dict.pop("processed_at", None)  # Remove column
            status = entry_dict.pop(
                "status", None
            )  # Pop and later push back to fix column order

            try:
                dbutils.fs.head(image_upload_path)  # type: ignore # noqa: F405

                image = OutputImage(image_upload_path)
                if self.annotate_images:
                    bounding_box = SilverDetectionMetadataManager.get_bounding_box_from_detection_id(
                        detection_id
                    )
                    image.draw_bounding_box(*bounding_box)

                notification_json = self.fill_incident_details(
                    incident_date=date_of_notification,
                    lon=LON,
                    lat=LAT,
                    object_class_str=object_class_str,
                )
                signal_id = self.post_signal_with_image_attachment(
                    json_content=notification_json, image=image
                )
                print(
                    f"Created signal {signal_id} for detection {detection_id} on {date_of_notification} with lat {LAT} and lon {LON}.\n\n"
                )
                entry_dict["signal_id"] = signal_id
                entry_dict["status"] = status  # Should be last column, so add it last
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
