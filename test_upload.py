import requests

url = "http://localhost:8000/api/upload/"
files = {'file': open('sample.txt', 'rb')}

response = requests.post(url, files=files)
print(response.status_code)
print(response.json())
