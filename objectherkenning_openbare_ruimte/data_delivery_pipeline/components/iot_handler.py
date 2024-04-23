import os
from contextlib import contextmanager
from typing import Dict

from azure.core.exceptions import AzureError
from azure.iot.device import IoTHubDeviceClient, Message
from azure.iot.device.common.models import X509
from azure.storage.blob import BlobClient
from azure.core.credentials import AccessToken


class AccessTokenCredential:
    def __init__(self, access_token):
        self._access_token = access_token

    def get_token(self, *scopes, **kwargs):
        return AccessToken(token=self._access_token, expires_on=None)


class IoTHandler:
    def __init__(
        self,
        hostname: str,
        device_id: str,
        shared_access_key: str = None,
        cert_file_path: str = None,
        key_file_path: str = None,
        passphrase: str = None,
    ):
        """
        Object that handles the connection with IoT and the delivery of messages and files.

        Parameters
        ----------
        hostname
            Azure IoT hostname
        device_id
            Device id of the device connecting
        shared_access_key
            Access key to authenticate
        """
        if shared_access_key:
            self.connection_string = (
                f"HostName={hostname};"
                f"DeviceId={device_id};"
                f"SharedAccessKey={shared_access_key}"
            )
        else:
            self.cert_file_path = cert_file_path
            self.key_file_path = key_file_path
            self.passphrase = passphrase
            self.hostname = hostname
            self.device_id = device_id

    @contextmanager
    def _connect(self):
        """
        Connects to Azure IoT.

        Example
        -------
        with self._connect() as device_client:
            device_client.send_message(message)
        """
        # x509 = X509(
        #     cert_file=self.cert_file_path,
        #     key_file=self.key_file_path,
        #     pass_phrase=self.passphrase,
        # )
        #
        # device_client = IoTHubDeviceClient.create_from_x509_certificate(
        #     x509=x509,
        #     hostname=self.hostname,
        #     device_id=self.device_id,
        #     websockets=True,
        # )
        device_client = IoTHubDeviceClient.create_from_connection_string(
            self.connection_string,
            websockets=True,
        )
        try:
            device_client.connect()
            yield device_client
        finally:
            device_client.shutdown()

    def deliver_message(self, message_content: str):
        """
        Delivers a message to Azure IoT.

        Parameters
        ----------
        message_content
            Content of the message.
        """
        message = Message(
            message_content, content_encoding="utf-8", content_type="application/json"
        )
        with self._connect() as device_client:
            device_client.send_message(message)

    def upload_file(self, file_path: str):
        """
        Uploads a file to Azure IoT.

        Parameters
        ----------
        file_path
            Path of the file to upload.
        """
        with self._connect() as device_client:
            blob_name = os.path.basename(file_path)
            storage_info = device_client.get_storage_info_for_blob(blob_name)

            success, result = self._store_blob(storage_info, file_path)
            if success:
                print(f"Upload succeeded. Result is: {result}")
                device_client.notify_blob_upload_status(
                    storage_info["correlationId"], True, 200, "OK: {}".format(file_path)
                )
            else:
                print(f"Upload failed. Exception is: {result}")
                device_client.notify_blob_upload_status(
                    storage_info["correlationId"],
                    False,
                    result.status_code,
                    str(result),
                )

    @staticmethod
    def _store_blob(blob_info: Dict[str, str], file_name: str):
        """
        Stores file on a blob container.

        Parameters
        ----------
        blob_info
            Dictionary containing: hostname, containername, blobname, and sastoken.
        file_name
            Path of the file to store
        """
        try:
            sas_url = "https://{}/{}/{}{}".format(
                blob_info["hostName"],
                blob_info["containerName"],
                blob_info["blobName"],
                blob_info["sasToken"],
            )

            print(
                "\nUploading file: {} to Azure Storage as blob: {} in container {}\n".format(
                    file_name, blob_info["blobName"], blob_info["containerName"]
                )
            )

            cred = AccessTokenCredential(blob_info["sasToken"].replace("?access_token=", ""))

            blob_client = BlobClient(blob_info["hostName"], credential=cred, container_name=blob_info["containerName"],
                                     blob_name=blob_info["blobName"])
            # with BlobClient.from_blob_url(sas_url) as blob_client:
            with open(file_name, "rb") as f:
                print(f)
                result = blob_client.upload_blob(f, overwrite=True)
                return True, result

            # Upload the specified file
            # with BlobClient.from_blob_url(sas_url) as blob_client:
            #     with open(file_name, "rb") as f:
            #         result = blob_client.upload_blob(f, overwrite=True)
            #         return True, result

        except FileNotFoundError as ex:
            return False, ex

        except AzureError as ex:
            return False, ex
