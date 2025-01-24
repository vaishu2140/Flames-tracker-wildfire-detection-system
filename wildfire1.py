import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from streamlit_option_menu import option_menu  # Install this package using 'pip install streamlit-option-menu'

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('wildfire.h5')  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
IMG_SIZE = (128, 128)  # Same size used during training

def preprocess_image(image):
    try:
        img = image.resize(IMG_SIZE)  # Resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Custom CSS for dynamic title and design
st.markdown(
    """
    <style>
        .fire-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            background: -webkit-linear-gradient(45deg, #FF4500, #FFD700, #FF6347);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subheader {
            text-align: center;
            color: #FF6347;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 50px;
            color: #AAAAAA;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navbar/Menu
selected = option_menu(
    menu_title=None,  # No navbar title
    options=["Home", "About", "Contact"],  # Menu options
    icons=["house", "info-circle", "envelope"],  # Icons for menu options
    menu_icon="cast",  # Menu icon
    default_index=0,  # Default active menu
    orientation="horizontal",  # Horizontal menu
    styles={
        "container": {"padding": "0!important", "background-color": "#f9f9f9"},
        "icon": {"color": "#FF6347", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#FFD700"},
    },
)

# Sidebar Menu Bar
st.sidebar.title("Menu Bar")
menu_selection = st.sidebar.radio(
    "Choose an Option:",
    ["Upload Image", "Prediction History", "System Info"]
)

if menu_selection == "Upload Image":
    st.sidebar.info("Upload an image for wildfire detection.")
elif menu_selection == "Prediction History":
    st.sidebar.info("Check your recent predictions.")
elif menu_selection == "System Info":
    st.sidebar.info("View details about this application.")

# Home Page
if selected == "Home":
    st.markdown('<div class="fire-title">🔥 Wildfire Detection System 🔥</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Classify satellite images into wildfire and no wildfire</div>', unsafe_allow_html=True)

    # Add wildfire-themed GIF
    st.image(
        "https://cdn.dribbble.com/users/2101543/screenshots/16170352/fire_dribble.gif", 
        caption="Stay Safe! Let's Detect Wildfires Early.", 
        use_container_width=True
    )

    # Upload image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Processing Image..."):
                try:
                    # Preprocess and predict
                    image = load_img(uploaded_file)
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        prediction = model.predict(processed_image)[0][0]  # Extract single prediction value

                        # Display prediction
                        threshold = 0.5  # Adjust the threshold as needed
                        if prediction > threshold:
                            st.error(f"🔥**Wildfire Detected**🔥")
                            st.warning("Take action immediately to prevent spread!")
                        else:
                            st.success("✅ No Wildfire Detected")
                            st.balloons()
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# About Page
elif selected == "About":
    st.markdown('<div class="fire-title">About Wildfire Detection System</div>', unsafe_allow_html=True)
    st.write("""
        This Wildfire Detection System is a machine learning-powered tool designed to classify satellite images 
        into categories of wildfire and no wildfire. Early detection of wildfires can help in mitigating their 
        devastating impact on the environment and human lives.
    """)
    st.image("https://media.giphy.com/media/3o7abldj0b3rxrZUxW/giphy.gif", use_container_width=True)

# Contact Page
elif selected == "Contact":
    st.markdown('<div class="fire-title">Contact Us</div>', unsafe_allow_html=True)
    st.write("""
        - **Email**: wildfire.detection@example.com  
        - **Phone**: +1 234 567 890  
        - **Address**: 123 Wildfire Prevention Lane, Earth
    """)
    st.image("https://media.giphy.com/media/26ufcYAKYe9qzdXX2/giphy.gif", use_container_width=True)

# Footer section
st.markdown(
    
    unsafe_allow_html=True,
)
