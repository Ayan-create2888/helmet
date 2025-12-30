import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Helmet Detection", page_icon="🪖")
st.title("🪖 Helmet Detection")

# Load the helmet detection model
@st.cache_resource(show_spinner=True)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_helmet.pt")
    return YOLO(model_path)

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # IMAGE PROCESSING
    if suffix in [".jpg", ".jpeg", ".png"]:
        st.image(Image.open(temp_file.name), caption="Original Image", use_column_width=True)

        # Run helmet detection
        results = model.predict(temp_file.name, save=True)
        output_path = results[0].save_dir + "/" + os.path.basename(temp_file.name)

        st.image(output_path, caption="Helmet Detection Result", use_column_width=True)

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇ Download Result",
                data=f,
                file_name="helmet_detection.jpg",
                mime="image/jpeg"
            )

    # VIDEO PROCESSING
    else:
        st.video(temp_file.name)

        # Run helmet detection on video
        results = model.predict(temp_file.name, save=True)
        output_video = results[0].save_dir + "/" + os.path.basename(temp_file.name)

        st.video(output_video)

        with open(ou
