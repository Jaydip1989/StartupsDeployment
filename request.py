import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={
	'R&D Spend':170000,
	'Administration': 120010,
	'Marketing Spend': 450000,
	'State_California':0,
	'State_Florida':0,
	'State_New York':1
	})