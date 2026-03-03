import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime
from logger import save_result

st.set_page_config(
    page_title="Driver Safety Monitoring System",
    page_icon="🚘",
    layout="centered"
)

# ----------- Custom CSS -----------
st.markdown("""
<style>
/* Full white background */
body {
    background-color: #ffffff;
}

/* Reduce page width and center nicely */
.block-container {
    max-width: 650px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title styling */
.main-title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    color: #0f172a;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #64748b;
    margin-bottom: 30px;
}

/* Card style container */
.result-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    text-align: center;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(to right, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    height: 48px;
    width: 220px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #1d4ed8, #1e40af);
}
</style>
""", unsafe_allow_html=True)

# ----------- Header -----------
st.markdown('<div class="main-title">Driver Safety Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Based Driver Behavior & Complaint Analysis</div>', unsafe_allow_html=True)

# ----------- Load Models -----------
image_model = tf.keras.models.load_model(
    "image_model.h5",
    compile=False
)
text_model = pickle.load(open("text_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ----------- Inputs -----------
uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "png", "jpeg"])
review = st.text_area("Enter Customer Complaint")

st.markdown("<br>", unsafe_allow_html=True)

# ----------- Analyze Button -----------
if st.button("Run Safety Analysis"):

    if uploaded_file is not None and review.strip() != "":

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Uploaded Driver Image", width=300)

        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        classes = ['other_activities','safe','talking_phone','texting_phone','turning']
        driver_status = classes[class_index]

        review_vector = vectorizer.transform([review])
        sentiment = text_model.predict(review_vector)[0]
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_result(driver_status, sentiment, confidence, timestamp)

        st.markdown("### Analysis Summary")

        st.write("Driver Behavior:", driver_status.capitalize())
        st.write("Complaint Sentiment:", sentiment.capitalize())
        st.write(f"Model Confidence: {confidence:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        if driver_status != "safe" or sentiment == "negative":
            st.error("⚠ Safety Status: Unsafe Driver Detected")
        else:
            st.success("✅ Safety Status: Driver is Safe")

        st.caption(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.markdown('</div>', unsafe_allow_html=True)

    else:

        st.warning("Please upload driver image and enter complaint text.")
