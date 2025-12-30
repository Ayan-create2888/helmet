import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import tempfile
import os

st.set_page_config(page_title="Helmet Detection", page_icon="🪖")
st.title("🪖 Helmet Detection")

@st.cache_resource(show_spinner=True)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_helmet.pt")
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # IMAGE
    if suffix in [".jpg", ".jpeg", ".png"]:
        st.image(Image.open(temp_file.name), caption="Original Image", use_column_width=True)

        results = model.predict(temp_file.name, save=True)
        # Ultralytics YOLO saves images in results[0].save_dir / filename
        output_path = os.path.join(results[0].save_dir, os.path.basename(temp_file.name))

        # Read the image into memory
        with open(output_path, "rb") as f:
            img_bytes = f.read()

        st.image(BytesIO(img_bytes), caption="Helmet Detection Result", use_column_width=True)
        st.download_button(
            label="⬇ Download Result",
            data=img_bytes,
            file_name="helmet_detection.jpg",
            mime="image/jpeg"
        )

    # VIDEO
    else:
        st.video(temp_file.name)

        results = model.predict(temp_file.name, save=True)
        output_video = os.path.join(results[0].save_dir, os.path.basename(temp_file.name))

        with open(output_video, "rb") as f:
            video_bytes = f.read()

        st.video(BytesIO(video_bytes))
        st.download_button(
            label="⬇ Download Result Video",
            data=video_bytes,
            file_name="helmet_detection.mp4",
            mime="video/mp4"
        )
