import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

MODEL_PATH = r"D:\Documents\University\Adv Proj\Train Model\YOLO\V3.1\yolo_driver_v3.1\weights\best.pt"

st.title("Driver Phone Usage Detection")
st.write("Upload a video file to run the YOLOv8 model.")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))
    
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB")
        
    cap.release()
    out.release()
    st.success("Processing Complete")
    
    with open(out_file.name, 'rb') as f:
        st.download_button("Download Processed Video", f, file_name="Predicted.mp4")