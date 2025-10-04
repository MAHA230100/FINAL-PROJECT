from __future__ import annotations

import time
import requests

API = "http://localhost:8000/health"
UI = "http://localhost:8501"

for i in range(20):
	try:
		r = requests.get(API, timeout=2)
		if r.ok:
			print("API health:", r.json())
			break
	except Exception:
		pass
	time.sleep(1)
else:
	print("API not responding at", API)

print("Visit UI:", UI) 