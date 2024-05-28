# Python In-built packages
from pathlib import Path
from PIL import Image
import joblib
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import torchvision.models as models
import torch.nn as nn
import torch
import cv2

def predict(img, model):
    img = np.array(img)
    img = cv2.resize(img, (64, 64))
    image = img.astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)

    outputs = model(image)
    # return values, indices
    _, predicted = torch.max(outputs, 1)
    return predicted

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
model_name = st.sidebar.radio("Select Model", ['MobileNet', 'ShuffleNet', 'EfficientNet'])


@staticmethod
def load_model():
    #'Mobile':
    mobilenet = models.mobilenet_v3_small(pretrained=True)
    mobilenet.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features=576, out_features=2, bias=True)
    )
    mobilenet.load_state_dict(torch.load('mobilenet.pt', map_location='cpu'))
    # 'EfficientNet':
    efficientnet = models.efficientnet_b0(pretrained=True)
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=2, bias=True)
    )
    efficientnet.load_state_dict(torch.load('efficientnet.pt', map_location='cpu'))
    # 'ShuffleNet':
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    shufflenet.fc = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=2, bias=True)
    )
    shufflenet.load_state_dict(torch.load('shufflenet.pt', map_location='cpu'))
    return mobilenet.eval(), efficientnet.eval(), shufflenet.eval()
    
mobilenet, efficientnet, shufflenet = load_model()

# Selecting Detection Or Segmentation
if model_name == 'MobileNet':
    model = mobilenet
elif model_name == 'EfficientNet':
    model = efficientnet
elif model_name == 'ShuffleNet':
    model = shufflenet



# Load Pre-trained ML Model
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", ['Image', 'Webcam Image', 'Webcam Video'])

prediction = None
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
                    prediction = predict(uploaded_image, model)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)
    with col2:
        if prediction is not None:
            st.write('\n\n\n\n\n\n\n\n\n')
            if prediction:
                st.header('This is a real face.')
            else:
                st.header('This is a fake face.')
        

elif source_radio == 'Webcam Image':
    col1, col2 = st.columns(2)
    with col1:
        image = st.camera_input("Take a picture")
        if image and st.button('PREDICT'):
            try:
                webcam_image = Image.open(image)
                prediction  = predict(webcam_image, model)
            except Exception as ex:
                st.error("Error occurred while processing the webcam image.")
                st.error(ex)
    with col2:
        if prediction is not None:
            st.write('\n\n\n\n\n\n\n\n\n')
            if prediction:
                st.header('This is a real face.')
            else:
                st.header('This is a fake face.')

# elif source_radio == 'Webcam Video':
#     FRAME_WINDOW = st.image([])
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
#             FRAME_WINDOW.image(webcam_frame)
#         with col2:
#             if predict is not None:
#                 st.write('\n\n\n\n\n\n\n\n\n')
#                 if predict:
#                     st.header('This is a real face.')
#                 else:
#                     st.header('This is a fake face.')
# else:
#     st.error("Please select a valid source type!")
