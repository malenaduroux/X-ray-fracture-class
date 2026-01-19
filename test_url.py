import requests

url = 'http://localhost:8080/predict'

request = {
    "url": "https://github.com/malenaduroux/X-ray-fracture-class/blob/main/test_image_fractured.jpg"
}

response = requests.post(url, json=request)
result = response.json()

print("STATUS:", response.status_code)
print("TEXT:", response.text)

try:
    result = response.json()
    print(result)
except Exception as e:
    print("JSON ERROR:", e)