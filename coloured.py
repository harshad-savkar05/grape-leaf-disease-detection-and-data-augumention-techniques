import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Function to create the model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img_array = np.frombuffer(uploaded_image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Error: Unable to load the image. Please select a valid image file.")
        return None
    
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)
    return img, img_array

# Function to predict the uploaded image
def predict_uploaded_image(model, uploaded_image, categories, top_n=4):
    try:
        img, img_array = preprocess_image(uploaded_image)
        if img is None:
            return
        
        predictions = model.predict(img_array)[0]
        top_indices = np.argsort(predictions)[::-1]
        top_predictions = [(categories[i], predictions[i] * 100) for i in top_indices[:top_n]]

        st.write("\n### Prediction Summary:")
        for rank, (category, confidence) in enumerate(top_predictions, start=1):
            st.write(f"{rank}. {category}: {confidence:.2f}%")

        # Plot confidence bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(categories, predictions * 100, color='skyblue')
        ax.set_xlabel('Classes', fontsize=14)
        ax.set_ylabel('Confidence (%)', fontsize=14)
        ax.set_title('Class Probability Distribution', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Show image with predictions
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        title_text = "\n".join([f"{rank}. {category}: {confidence:.2f}%" for rank, (category, confidence) in enumerate(top_predictions, start=1)])
        ax.set_title(title_text, fontsize=12)
        st.pyplot(fig)

        suggest_fertilizer(top_predictions[0][0])

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Suggest fertilizer based on disease
def suggest_fertilizer(disease):
    fertilizer_suggestions = {
        "Black_rot": (
            "Apply a copper-based fungicide like Bordeaux mixture. "
            "Use balanced NPK fertilizer (10-10-10) and ensure proper drainage. "
            "Prune infected areas and maintain good air circulation."
        ),
        "Esca_(Black_Measles)": (
            "Use systemic fungicides such as thiophanate-methyl or mancozeb. "
            "Incorporate phosphorus-rich fertilizers (e.g., 10-20-10) to support root development. "
            "Avoid overwatering and remove affected wood parts."
        ),
        "Healthy": (
            "No chemical treatment needed. Maintain optimal health using compost or organic manure. "
            "Apply micronutrients (like Zn, Fe, Mg) periodically to prevent deficiencies."
        ),
        "Leaf_blight_(Isariopsis_Leaf_Spot)": (
            "Use DMI-based fungicides (e.g., tebuconazole) during early disease stages. "
            "Apply potassium-rich fertilizer (e.g., 0-0-50) to strengthen leaves and improve stress resistance. "
            "Avoid wet foliage during irrigation."
        ),
        "Anthracnose": (
            "Spray fungicide like chlorothalonil or copper oxychloride. "
            "Use calcium and potassium-enriched fertilizers to strengthen tissue. "
            "Keep canopy open to reduce humidity."
        ),
        "Downy_Mildew": (
            "Use metalaxyl or fosetyl-Al fungicides. "
            "Apply nitrogen-based fertilizer early in the growing season (e.g., 20-10-10). "
            "Ensure proper drainage and avoid waterlogging."
        ),
        "Powdery_Mildew": (
            "Apply sulfur-based fungicide or potassium bicarbonate sprays. "
            "Use balanced fertilizer with lower nitrogen to prevent excessive soft growth. "
            "Spray neem oil for organic management."
        )
    }

    suggestion = fertilizer_suggestions.get(disease, "Fertilizer suggestion not available for this disease.")
    st.write(f"### Recommended Fertilizer and Treatment:\n{suggestion}")

# Preprocess image data from folders
def preprocess_data(data_dir, categories):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            st.warning(f"Folder for {category} does not exist. Skipping...")
            continue
        class_label = categories.index(category)
        for img_file in os.listdir(path):
            try:
                img_path = os.path.join(path, img_file)
                img_array = cv2.imread(img_path)
                img_resized = cv2.resize(img_array, (256, 256))
                data.append([img_resized, class_label])
            except Exception as e:
                print(f"Error loading image: {e}")
    random.shuffle(data)
    X, Y = zip(*data)
    return np.array(X), np.array(Y)

# Model training function
def train_model(model, x_train, y_train, x_val, y_val, epochs=10):
    y_train_cat = to_categorical(y_train, 4)
    y_val_cat = to_categorical(y_val, 4)
    history = model.fit(x_train, y_train_cat, validation_data=(x_val, y_val_cat),
                        epochs=epochs, batch_size=32, verbose=1)
    return history

# Webcam video transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model('leaf_disease_model.h5')
        self.categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_array = np.expand_dims(img_normalized, axis=0)

        predictions = self.model.predict(img_array)[0]
        top_indices = np.argsort(predictions)[::-1]
        top_predictions = [(self.categories[i], predictions[i] * 100) for i in top_indices[:4]]

        for rank, (category, confidence) in enumerate(top_predictions, start=1):
            cv2.putText(img, f"{category}: {confidence:.2f}%", (10, 30 + rank * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app main interface
def main():
    st.title("🍇 Grape Leaf Disease Detection")
    st.write("Upload a grape leaf image or scan live using your webcam.")

    uploaded_image = st.file_uploader("📷 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_container_width=True)
        st.write("🔍 Classifying...")
        try:
            model = load_model('leaf_disease_model.h5')
            categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]
            predict_uploaded_image(model, uploaded_image, categories)
        except Exception as e:
            st.write(f"❌ Error: {e}")
    
    # Webcam live detection
    st.write("---")
    st.subheader("📹 Real-time Webcam Detection")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
