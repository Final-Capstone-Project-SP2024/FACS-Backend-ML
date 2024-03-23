import requests

detect_url = "https://firealarmcamerasolution.azurewebsites.net/api/v1/Camera/ae345580-8276-4181-ba97-33f95df700f9/detect"
# camera_id = 'ae345580-8276-4181-ba97-33f95df700f9'

data = {
    "predictedPercent": 100,
    "pictureUrl": "1",
    "videoUrl" : "video_url"
}

headers = {"x-api-key": "comsuonhocmon"}

response = requests.post(detect_url, json=data, headers=headers)

if response.status_code == 200:
    print("POST request successful.")
else:
    print(f"Failed to send POST request. Status code: {response.status_code}")