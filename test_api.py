import json
import urllib.request

# Hamari API ka URL
url = 'http://127.0.0.1:5000/predict'

# 19 features ka dummy data (kuch values badi rakhi hain taaki Fraud detect ho aur XAI chale)
data = {
    "features": [0.9, 0.8, 75000.0, 2.5, 0.1, 0.8, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}

# Data ko JSON format me API ko bhejna
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

print("Sending data to API...\n")
try:
    response = urllib.request.urlopen(req)
    result = json.loads(response.read().decode('utf-8'))
    
    # Output ko sundar tarike se print karna
    print("--- AUN'S API OUTPUT ---")
    print(json.dumps(result, indent=4))
    
except Exception as e:
    print("Error aaya:", e)