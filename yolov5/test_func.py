import requests

url = "https://firealarmcamerasolution.azurewebsites.net/api/v1/Camera"
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1lIjoic3RyaW5nIiwiaHR0cDovL3NjaGVtYXMubWljcm9zb2Z0LmNvbS93cy8yMDA4LzA2L2lkZW50aXR5L2NsYWltcy9yb2xlIjoiTWFuYWdlciIsIlVzZXJJZCI6IjBmMzhlYjEwLTFhZTQtNDlhMS1iMWUzLTVkMTUzYjIzOThiMCIsImV4cCI6MTcxNjM1NDQyNCwiaXNzIjoiRkFDUyAtIEZpcmUgQWxhcm0gQ2FtZXJhIFNvbHV0aW9uIiwiYXVkIjoiRkFDUyAtIEZpcmUgQWxhcm0gQ2FtZXJhIFNvbHV0aW9uIn0.qbcARNPtK5fcBBmbNCM-MttM0ou3btbj3G7LdFcRVzI"

headers = {
    "Authorization": f"Bearer {jwt_token}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print("Failed to retrieve data. Status code:", response.status_code)
