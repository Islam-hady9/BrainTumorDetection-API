import requests

url = "http://127.0.0.1:8000/predict/"
image_path = "Model/Test Images/Y01.JPG"

with open(image_path, "rb") as image_file:
    response = requests.post(url, files={"file": image_file})
    print(response.json())