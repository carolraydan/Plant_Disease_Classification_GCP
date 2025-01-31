import streamlit as st
import requests
from PIL import Image
import io

# URL of your GCP container service
GCP_API_URL = "https://model-h5-final-401399505381.us-central1.run.app/predict"  # Replace with your actual URL

# Function to call the GCP container's API endpoint
def call_gcp_container(image_data):
    # Preparing the image to be sent with the correct form-data format
    files = {"files": ("image", image_data, "image/jpeg")}  # Key is 'files', value is the image content with mime type
    
    # Sending a POST request to the GCP container with the image data
    response = requests.post(GCP_API_URL, files=files)
    
    if response.status_code == 200:
        return response.json()  # Assuming the response is in JSON format
    else:
        st.error(f"Request failed with status code {response.status_code}: {response.text}")
        return None

# Streamlit UI
st.set_page_config(page_title="Potato Plant Disease Classification", page_icon="🌱", layout="wide")
st.title("🌿 Potato Plant Disease Classification 🌿")

# Adding custom background and text styling
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
        }
        .stFileUploader {
            border-radius: 12px;
            border: 2px solid #ccc;
            padding: 10px;
        }
        .stTextInput>div>input {
            border-radius: 8px;
            padding: 10px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        
        /* CSS to center image and result */
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .image {
            max-width: 60%;  /* Resize the image */
            height: auto;
        }

        /* Centered result container */
        .result-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 60%;
            margin-left: auto;
            margin-right: auto;
            margin-top: 20px;  /* Add margin-top for space between result and image */
        }

        .prediction-title {
            color: black;
            font-size: 20px;
        }

        .prediction-result {
            color: black;  /* Ensure text is black */
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

st.write("Upload an image of a potato plant to get predictions.")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image with enhanced styling and centered
    image = Image.open(uploaded_file)
    
    # Resize the image to 224x224
    image = image.resize((224, 224))
    
    # Center the image using CSS flexbox
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", width=400)  # Set a fixed width for the image
    st.markdown('</div>', unsafe_allow_html=True)

    # Convert the resized image to byte data (this is what the API expects)
    image_byte_data = io.BytesIO()  # Create a BytesIO object to hold the byte data
    # Convert the resized image to byte data (this is what the API expects)
    image_byte_data = io.BytesIO()  # Create a BytesIO object to hold the byte data

    # Convert image to RGB mode if it has an alpha channel (RGBA)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    # Save the image to the byte buffer
    image.save(image_byte_data, format="JPEG")
    image_byte_data.seek(0)  # Move to the start of the BytesIO buffer
    image_data = image_byte_data.read()  # Read the byte data

    # Call GCP container API
    st.write("Calling the GCP container... Please wait.")
    result = call_gcp_container(image_data)

    if result:
        st.markdown("""
        <div class="result-container">
            <h3 class="prediction-title"></h3>
            <p class="prediction-result"><strong>Class:</strong> {}</p>
            <p class="prediction-result"><strong>Confidence:</strong> {:.2f}%</p>
        </div>
        """.format(result['class'], result['confidence'] * 100), unsafe_allow_html=True)
