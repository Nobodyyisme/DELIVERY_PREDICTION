import base64
import time
import pickle
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import xgboost

# Set the page configuration of the app, including the page title, icon, and layout.
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# Load your local image and convert it to base64
img_base64 = get_base64_image("image2.jpg")

# Inject CSS with the base64 image as the background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}"); /* Base64 image as background */
        background-size: cover;
        background-position: top center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        height: 100vh; /* Full screen height */
        width: 100%;
        color: rgba(255, 255, 255, 0.8);
    }}
    
    
    .stContainer {{
        background: rgba(255, 255, 255, 0.3); /* Transparent white background */
        backdrop-filter: blur(10px); /* Apply the blur effect */
        border-radius: 15px; /* Rounded corners */
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Soft shadow for the glass effect */
        margin: 20px; /* Margin around the glass container */
        max-width: 1200px; /* Maximum width of the content */
        margin-top: 100px; /* Space from top */
    }}
    
    .stSidebar {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(128, 128, 128, 0.3)); /* Metallic gradient effect */
        backdrop-filter: blur(15px); /* Apply blur effect */
        border-radius: 15px; /* Rounded corners */
        color: rgba(255, 255,255, 0.8);
        padding: 5px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4); /* Strong shadow for the metallic look */
        border: 1px solid rgba(255, 255, 255, 0.3); /* Subtle border to enhance the metallic effect */
    }}
    
    .stSidebar .sidebar-content {{
        color: #FFFFFF; /* White text color to contrast with metallic background */
    }}
    
    div[class*="stNumberInput"] label {{
        font-size: 35px;
        font-weight: bold;
        color: rgba(255, 255, 255, 1);
    }}
    
    
    /* Custom caption styling */
    .custom-caption {{
        text-align: justify; /* Justify the caption text */
        font-size: 16px; /* Optional: Adjust font size */
        color: rgba(255, 255, 255, 0.7); /* Yellow font color */
        line-height: 1.5; /* Add some line spacing for readability */
        font-weight: normal;
    }}
    
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title and captions for the app.
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

# Apply custom style for caption
st.markdown('<p class="custom-caption">Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations.</p>', unsafe_allow_html=True)
st.markdown('<p class="custom-caption">With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain.</p>', unsafe_allow_html=True)

if "df" not in st.session_state:
    # Define a sample dataset for demonstration purposes.
        data = {
            "Purchased Day of the Week": [],
            "Purchased Month": [],
            "Purchased Year": [],
            "Product Size in cm^3": [],
            "Product Weight in grams": [],
            "Geolocation State Customer": [],
            "Geolocation State Seller": [],
            "Distance": [],
            "wait_time":[]
        }
        # Create a DataFrame from the sample dataset.
        st.session_state.df = pd.DataFrame(data)



# Define the wait time prediction function (assuming model is cached)
@st.cache_resource
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
    df
):
    new_row = {
        "purchase_dow": purchase_dow,
        "purchase_month": purchase_month,
        "year": year,
        "product_size_cm3": product_size_cm3,
        "product_weight_g": product_weight_g,
        "geolocation_state_customer": geolocation_state_customer,
        "geolocation_state_seller": geolocation_state_seller,
        "distance": distance,
    }
    
    modelfile = "voting_model.pkl"
    voting_model = pickle.load(open(modelfile, 'rb'))  
    new_row = pd.DataFrame([new_row])# nose
    prediction = voting_model.predict(new_row)
    print(int(prediction[0]))

    add_row(purchase_dow,
        purchase_month,
        year,
        product_size_cm3,
        product_weight_g,
        geolocation_state_customer,
        geolocation_state_seller,
        distance,
        int(prediction[0])
    )

    return int(prediction[0])

def add_row(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
    wait_time
):
    new_row = {
        "Purchased Day of the Week": purchase_dow,
        "Purchased Month": purchase_month,
        "Purchased Year": year,
        "Product Size in cm^3": product_size_cm3,
        "Product Weight in grams": product_weight_g,
        "Geolocation State Customer": geolocation_state_customer,
        "Geolocation State Seller": geolocation_state_seller,
        "Distance": distance,
        "wait_time":wait_time
    }
    
   # st.write(df)
    df = st.session_state.df
    df.loc[len(df)] = new_row
    
# Define the input parameters using Streamlit's sidebar.
with st.sidebar:
    img = Image.open("image1.jpg")
    st.image(img)
    st.header("Input Parameters")
    purchase_dow = st.number_input(
        "Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3
    )
    purchase_month = st.number_input(
        "Purchased Month", min_value=1, max_value=12, step=1, value=1
    )
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input(
        "Geolocation State of the Customer", value=10
    )
    geolocation_state_seller = st.number_input(
        "Geolocation State of the Seller", value=20
    )
    distance = st.number_input("Distance", value=475.35)
    submit = st.button(label="Predict Wait Time!")
    #                     , on_click=lambda:add_row(purchase_dow,
    # purchase_month,
    # year,
    # product_size_cm3,
    # product_weight_g,
    # geolocation_state_customer,
    # geolocation_state_seller,
    # distance
    # ))
    
# Define the submit button for the input parameters.
with st.container():
    # Define a sample dataset for demonstration purposes.
       # Display the sample dataset in the Streamlit app.
    if submit:
        prediction = waitime_predictor(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
            st.session_state.df
        )
        st.markdown("Output: Wait Time in Days")
        st.markdown(f"<h1 style='font-size: 40px; color: white;'>Output: Wait Time in Days</h1>", unsafe_allow_html=True)
        with st.spinner(text="This may take a moment..."):
            st.markdown(f"<h1 style='font-size: 40px; color: white;'>{prediction} days</h1>", unsafe_allow_html=True)

        
        st.header("Past Dataset")
        st.write(st.session_state.df)
        
    # Define the output container for the predicted wait time.

    # When the submit button is clicked, call the wait time predictor function and display the predicted wait time in the output container.

