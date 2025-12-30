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

        results = model.predict(temp_file.name, save=False)  # no need to save on disk
        annotated_img = results[0].plot()  # returns NumPy array
        img_pil = Image.fromarray(annotated_img)

        img_bytes = BytesIO()
        img_pil.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        st.image(img_bytes, caption="Helmet Detection Result", use_column_width=True)
        st.download_button(
            label="⬇ Download Result",
            data=img_bytes,
            file_name="helmet_detection.jpg",
            mime="image/jpeg"
        )

    # VIDEO
    else:
        st.video(temp_file.name)

        results = model.predict(temp_file.name, save=True)  # videos must be saved
        # Safest: get actual saved video path from results
        output_video_path = results[0].path  # path is guaranteed to exist

        with open(output_video_path, "rb") as f:
            video_bytes = f.read()

        st.video(BytesIO(video_bytes))
        st.download_button(
            label="⬇ Download Result Video",
            data=video_bytes,
            file_name="helmet_detection.mp4",
            mime="video/mp4"
        )
