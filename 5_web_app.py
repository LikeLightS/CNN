import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# = Title =
st.set_page_config(page_title="Casting Inspection", layout="wide")
st.title("Casting Inspection using CNN")
st.write("Upload an image of a metal casting from https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product to see the neural network predictions.")

# = Load model ==
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("casting_defect_model.pth", map_location = device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()


# = Image selection =
st.write("### Sample Images")
SAMPLE_IMAGE = {
    "Select a sample..." : "",
    "Sample 1: Defective" : "casting_data/test/def_front/cast_def_0_15.jpeg",
    "Sample 2: Defective" : "casting_data/test/def_front/cast_def_0_61.jpeg",
    "Sample 3: Defective" : "casting_data/test/def_front/cast_def_0_218.jpeg",
    "Sample 4: OK" : "casting_data/test/ok_front/cast_ok_0_235.jpeg",
    "Sample 5: OK" : "casting_data/test/ok_front/cast_ok_0_942.jpeg"
}

uploaded_image = st.file_uploader("Upload casting image (JPG/PNG)...", type=["JPEG", "JPG", "PNG"])
selected_sample_image = st.selectbox("Choose a pre-loaded sample...", list(SAMPLE_IMAGE.keys()))
image_to_process = None

if uploaded_image is not None:
    image_to_process = Image.open(uploaded_image).convert("RGB").resize((224,224))
elif selected_sample_image != "Select a sample...":
    sample_path = SAMPLE_IMAGE[selected_sample_image]
    image_to_process = Image.open(sample_path).convert("RGB").resize((224,224))

if image_to_process is None: # Default
    image_to_process = Image.open("casting_data/test/def_front/cast_def_0_15.jpeg").convert("RGB").resize((224,224))

# = Predictions =
test_transform = transforms.Compose([transforms.ToTensor()])
input_tensor = test_transform(image_to_process).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

class_name = ["Defective", "OK"]
prediction_label = class_name[prediction]


# = Heatmap =
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(0)]
grayscale_cam = cam(input_tensor = input_tensor, targets=targets)[0,:]
rgb_img = np.float32(image_to_process) / 255.0
visualize = show_cam_on_image(rgb_img, grayscale_cam, True)


# = Display =
st.divider()

if prediction == 0:
    st.error(f"Prediction: {prediction_label}")
else:
    st.success(f"Prediction: {prediction_label}")

col1, col2 = st.columns(2)

with col1:
    st.header("Original Image")
    st.image(image_to_process)
with col2:
    st.header("GradCAM heatmap image")
    st.image(visualize)
