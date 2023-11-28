import requests

# Define the URL of your Flask application's /predict endpoint
url = "https://my-app-capstone-403817.nw.r.appspot.com/"  # Replace with the actual URL where your Flask app is running


# Input values as a comma-separated string
input_values = "10.24,0.74,0.52,46741.5,3.5,478498.5,0.84,6.0,24199.15,454500.0,0.0,0.0,0.0,-17955.0,1.0,0.0,0.07,2016.0,6.0,0.0,1.0,4033.26,-1516.0,0.0,0.0,175500.0,0.0,49.19,-814.0,0.0,13.0,-634.0,0.0,6.0,87750.0"

# Create a dictionary with the input values
data = {"input_values": input_values}

# Send a POST request to the Flask application
response = requests.post(url, data=data)

# Check if the request was successful
if response.status_code == 200:
    result = response.text
else:
    result = f"Error: Request failed with status code {response.status_code}"

print(result)
