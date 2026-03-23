import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from emotion_detector import detect_emotion

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Emotion Detection AI", layout="wide", page_icon="😊")

# ==========================================
# 2. SIDEBAR & THEME SETTINGS
# ==========================================
st.sidebar.title("⚙️ Settings")
theme_option = st.sidebar.radio(
    "🎨 Select UI Theme",
    ("Dark Premium", "Light Minimalist")
)

option = st.sidebar.selectbox(
    "🔍 Select Detection Mode",
    ("Image Detection", "Video Detection")
)

# Dynamic Theme Variables
if theme_option == "Dark Premium":
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#161A24"
    border_color = "#2B2E3B"
    accent_primary = "#6B66FF"
    accent_secondary = "#FF6B6B"
    box_color = (255, 107, 107) # BGR
else:
    bg_color = "#F8FAFC"
    text_color = "#0F172A"
    card_bg = "#FFFFFF"
    border_color = "#E2E8F0"
    accent_primary = "#3B82F6"
    accent_secondary = "#10B981"
    box_color = (246, 130, 59) # BGR

# Inject Dynamic CSS
dynamic_css = f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {card_bg};
        border-right: 1px solid {border_color};
    }}
    h1, h2, h3, p {{
        color: {text_color} !important;
    }}
    h1 {{
        background: -webkit-linear-gradient(45deg, {accent_secondary}, {accent_primary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        padding-bottom: 10px;
    }}
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed {accent_primary} !important;
        background-color: {card_bg} !important;
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
    }}
    [data-testid="stFileUploadDropzone"]:hover {{
        border-color: {accent_secondary} !important;
    }}
</style>
"""
st.markdown(dynamic_css, unsafe_allow_html=True)

st.title("😊 AI Emotion Detection System")

# ==========================================
# 3. HELPER FUNCTIONS & CACHING
# ==========================================
@st.cache_resource
def load_cascade():
    # Cache the cascade so it doesn't load on every run
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_cascade()

def draw_chart(df, bar_color):
    fig, ax = plt.subplots(facecolor=bg_color, figsize=(8, 4))
    ax.set_facecolor(bg_color)
    df.mean().plot(kind="bar", ax=ax, color=bar_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(border_color)
    return fig

# ==========================================
# 4. IMAGE DETECTION
# ==========================================
if option == "Image Detection":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp", "bmp", "dib", "tiff", "tif", "jfif", "jp2"])

    if uploaded_image:
        with st.spinner("⚡ Processing Image..."):
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Speed Optimization: Shrink large images
            h, w = frame.shape[:2]
            max_width = 800
            if w > max_width:
                new_h = int(h * (max_width / w))
                frame = cv2.resize(frame, (max_width, new_h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
            emotion_data = []

            for (x,y,w,h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face,(224,224))
                emotion, emotions = detect_emotion(face)

                if emotion:
                    emotion_data.append(emotions)
                    cv2.rectangle(frame,(x,y),(x+w,y+h), box_color, 3)
                    cv2.putText(frame, emotion.upper(), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Centered Image Display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(frame, channels="BGR", width=500, caption="AI Processed Image")

        if emotion_data:
            st.markdown("---")
            df = pd.DataFrame(emotion_data)
            
            st.subheader("📊 Emotion Breakdown")
            st.dataframe(df, use_container_width=True)
            
            st.subheader("📈 Emotion Graph")
            st.pyplot(draw_chart(df, accent_primary))
        else:
            st.warning("No face detected in the image.")


# ==========================================
# 5. VIDEO DETECTION
# ==========================================
elif option == "Video Detection":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"])

    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_file.name)
        
        # Side-by-Side Video Layout
        st.markdown("---")
        vid_col1, vid_col2 = st.columns(2)
        
        with vid_col1:
            st.markdown(f"<h3 style='text-align: center; color: {text_color};'>📥 Original Input</h3>", unsafe_allow_html=True)
            st.video(uploaded_video) # Show the original uploaded video
            
        with vid_col2:
            st.markdown(f"<h3 style='text-align: center; color: {text_color};'>🤖 AI Processing</h3>", unsafe_allow_html=True)
            stframe = st.empty() # Placeholder for processed frames
            
        status_msg = st.empty()
        status_msg.info("⚡ Fast Processing mode active... Please wait.")
        progress_bar = st.progress(0)
            
        emotion_data = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = 10  # Process 1 out of 10 frames
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            if frame_count % 5 == 0 and total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            if frame_count % frame_skip != 0:
                continue

            h, w = frame.shape[:2]
            new_w = 480 
            new_h = int(h * (new_w / w))
            process_frame = cv2.resize(frame, (new_w, new_h))

            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))

            for (x,y,w,h) in faces:
                face = process_frame[y:y+h, x:x+w]
                face = cv2.resize(face,(224,224))
                emotion, emotions = detect_emotion(face)

                if emotion:
                    emotion_data.append(emotions)
                    cv2.rectangle(process_frame,(x,y),(x+w,y+h), box_color, 3)
                    cv2.putText(process_frame, emotion.upper(), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            # Display processed frame
            stframe.image(process_frame, channels="BGR", use_container_width=True)

        cap.release()
        progress_bar.empty()
        status_msg.success("✅ Video Processing Complete!")

        # Analytics below the videos
        if emotion_data:
            st.markdown("---")
            df = pd.DataFrame(emotion_data)
            
            st.subheader("📊 Overall Emotion Metrics")
            st.dataframe(df.describe(), use_container_width=True) 
            
            st.subheader("📈 Average Emotion Graph")
            st.pyplot(draw_chart(df, accent_secondary))
        else:
            st.warning("No faces detected in the video.")