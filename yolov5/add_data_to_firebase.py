import firebase_admin
from firebase_admin import credentials, storage
import datetime
from google.cloud import exceptions

def upload_file_to_storage(local_file_path, destination_file_name):
    if not firebase_admin._apps:
        cred = credentials.Certificate("./yolov5/serviceAccountKey.json") 
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'final-capstone-project-f8bdd.appspot.com'
        })

    bucket = storage.bucket()

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(local_file_path)

    try:
        blob.reload()
        url_expiration = datetime.timedelta(hours=1) 
        expiration_date = datetime.datetime.now() + url_expiration
        download_url = blob.generate_signed_url(expiration=expiration_date)
        return download_url
    except exceptions.NotFound:
        raise ValueError("The file upload was not successful. The blob doesn't exist in the bucket.")

if __name__ == "__main__":
    local_file_path = "./yolov5/inference/records/record_2024-03-23_09-16-34.avi.mp4"
    destination_file_name = "testkhongtroll.mp4"
    try:
        download_url = upload_file_to_storage(local_file_path, destination_file_name)
        print("File uploaded successfully. Download URL:", download_url)
    except ValueError as e:
        print(str(e))