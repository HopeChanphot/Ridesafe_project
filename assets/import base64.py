import base64

# Load the image and convert it to Base64
with open(r"C:\Users\user\Downloads\Bike_web_app\assets\RideSafe_logo.png", "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode('utf-8')

# Print the base64 string
print(base64_string)

