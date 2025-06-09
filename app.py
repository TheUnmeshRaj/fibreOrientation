import streamlit as st

st.set_page_config(page_title="Fiber Orientation Classifier", page_icon="🧵", layout="centered")

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

num_classes = 2  
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model1.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()
class_names = ['bidirectional','unidirectional']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        pred = torch.argmax(output, dim=1).item()
    return class_names[pred]

# --- Streamlit UI ---


st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #222222;
    }
    .stButton > button {
        background-color: #005f73;
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 150px;
        font-weight: 600;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0a9396;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧵 Fiber Orientation Classifier")
st.write("Upload an image to predict the fiber orientation category.")

uploaded_file = st.file_uploader("Choose an image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        label = predict(img)
    st.success(f"**Predicted fiber orientation:** {label}")
else:
    st.info("Please upload an image file to get started.")
