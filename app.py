# Python In-built packages
from pathlib import Path
from PIL import Image
import joblib
import cv2 as cv
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def calHist(img):
    hist = cv.calcHist([img],[0],None,
                       [256],[0,256])
    size = img.shape[0]*img.shape[1]
    hist = hist / size
    return hist.reshape(-1)


# Setting page layout
st.set_page_config(
    page_title="Deep Fake Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Deep Fake Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_name = st.sidebar.radio("Select Model", ['KNN', 'Decision Tree', 'Random Forest', 'Ensemble'])

# Selecting Detection Or Segmentation
if model_name == 'KNN':
    model = joblib.load('knn.pkl')
elif model_name == 'Decision Tree':
    model = joblib.load('decisiontree.pkl')
elif model_name == 'Random Forest':
    model = joblib.load('randomforest.pkl')
elif model_name == 'Ensemble':
    model = joblib.load('ensemble.pkl')



# Load Pre-trained ML Model
st.sidebar.header("Image Config")
source_radio = st.sidebar.radio("Select Source", ['Image', 'Webcam'])

predict = None
source_img = None
# If image is selected
if source_radio == 'Image':
    source_img = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)
    with col1:
        if source_img is not None:
            try:
                uploaded_image = Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
                if st.button('PREDICT'):
                    img = np.array(uploaded_image.convert('L'))
                    feature = calHist(img)
                    predict,  = model.predict([feature])
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)
    with col2:
        if predict is not None:
            st.write('\n\n\n\n\n\n\n\n\n')
            if predict:
                st.header('REAL FACE.')
            else:
                st.header('FAKE FACE')
        

elif source_radio == 'Webcam':
    col1, col2 = st.columns(2)
    with col1:
        image = st.camera_input("Take a picture")
        if image and st.button('PREDICT'):
            try:
                webcam_image = Image.open(image)
                img = np.array(webcam_image.convert('L'))
                feature = calHist(img)
                predict,  = model.predict([feature])
            except Exception as ex:
                st.error("Error occurred while processing the webcam image.")
                st.error(ex)
    with col2:
        if predict is not None:
            st.write('\n\n\n\n\n\n\n\n\n')
            if predict:
                st.header('REAL FACE.')
            else:
                st.header('FAKE FACE')

# elif source_radio == 'Webcam Video':
#     frame_placeholder = st.empty()
#     cap = cv.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             cap.release()
#             break
#         webcam_frame = Image.open(frame)
#         gray_frame = np.array(webcam_frame.convert('L'))
#         feature = calHist(gray_frame)
#         predict = model.predict([feature])[0]
#         col1, col2 = st.columns(2)
#         with col1:
#             frame_placeholder.image(webcam_frame)
#         with col2:
#             if predict is not None:
#                 st.write('\n\n\n\n\n\n\n\n\n')
#                 if predict:
#                     st.header('This is a real face.')
#                 else:
#                     st.header('This is a fake face.')
else:
    st.error("Please select a valid source type!")
