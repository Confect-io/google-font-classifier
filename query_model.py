import os

import requests

API_URL = "https://uyta9gge1o9rxpdq.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
	"Content-Type": "image/jpeg" 
}

def query(filename):
	with open(filename, "rb") as f:
		data = f.read()
	response = requests.post(API_URL, headers=headers, data=data)
	return response.json()

matches = 0
total = 0

for label in os.listdir("v4/dataset/test"):
	for image in os.listdir(f"v4/dataset/test/{label}"):
		output = query(f"v4/dataset/test/{label}/{image}")
		# [{"label": "Arial", "score": 0.9999999999999999}, {"label": "Times New Roman", "score": 0.0000000000000001}]

		output_label = output[0]["label"]

		if output_label == label:
			matches += 1
		else:
			print(f"Bad: v4/dataset/test/{label}/{image}")
		total += 1

		print(f"{matches}/{total}")
