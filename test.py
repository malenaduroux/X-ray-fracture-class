import requests

url = "http://localhost:8080/predict"

payload = {
    "url": "https://raw.githubusercontent.com/malenaduroux/X-ray-fracture-class/main/test_image_fractured.jpg"
}

response = requests.post(url, json=payload)

print("STATUS:", response.status_code)
print("TEXT:", response.text)

try:
    print(response.json())
except Exception as e:
    print("JSON ERROR:", e)