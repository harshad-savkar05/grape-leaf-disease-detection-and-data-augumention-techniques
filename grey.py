import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import streamlit as st
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Categories for classification
CATEGORIES = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]

# Function to create CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),  # Adjusted for grayscale input
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((8, 8)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((8, 8)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to upload an image via Streamlit
def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open the image using OpenCV
        img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (256, 256)) / 255.0  # Normalize the image
        img_array = img_resized.reshape(1, 256, 256, 1)  # Adjust to grayscale shape
        
        # Display the grayscale image
        st.image(img_resized, caption="Grayscale Image", use_container_width=True, channels="GRAY")
        
        return uploaded_file.name, img_array, img_resized
    else:
        return None, None, None

# Function to predict class of uploaded image
def predict_uploaded_image(model, img_array, file_path, gray_image):
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100

    st.write(f"Predictions for {file_path}:")
    for i, score in enumerate(predictions):
        st.write(f"{CATEGORIES[i]}: {score * 100:.2f}%")
    
    st.write(f"\n**Predicted Class**: {CATEGORIES[predicted_index]} with Confidence: {confidence:.2f}%")

    # Display the grayscale image and prediction
    st.image(gray_image, caption=f"Prediction: {CATEGORIES[predicted_index]} ({confidence:.2f}%)", use_container_width=True, channels="GRAY")

# Function to preprocess dataset
def preprocess_data(data_dir, categories):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                img_resized = cv2.resize(img, (256, 256)) / 255.0
                data.append([img_resized, class_num])
            except Exception as e:
                pass
    random.shuffle(data)
    x, y = zip(*data)
    x = np.array(x).reshape(-1, 256, 256, 1)  # Adjust to grayscale shape
    y = to_categorical(y, len(categories))
    return x, y

# Train and visualize model with epoch-wise accuracy and loss
def train_and_visualize(x_train, y_train, x_test, y_test):
    model = create_cnn_model()

    # Initialize lists to store accuracy and loss for each epoch
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # Train the model
    st.write("Training the model...")
    for epoch in range(10):
        history = model.fit(
            x_train, y_train,
            epochs=1, batch_size=32,
            validation_data=(x_test, y_test),  # Include validation data
            shuffle=True, verbose=1
        )

        # Store accuracy and loss for each epoch
        train_accuracies.append(history.history['accuracy'][0])
        val_accuracies.append(history.history['val_accuracy'][0])
        train_losses.append(history.history['loss'][0])
        val_losses.append(history.history['val_loss'][0])

        st.write(f"Epoch {epoch + 1}:")
        st.write(f"Train Accuracy: {history.history['accuracy'][0] * 100:.2f}%")
        st.write(f"Validation Accuracy: {history.history['val_accuracy'][0] * 100:.2f}%")
        st.write(f"Train Loss: {history.history['loss'][0]:.4f}")
        st.write(f"Validation Loss: {history.history['val_loss'][0]:.4f}")
        st.write("-" * 50)

    # Save the model
    model.save("leaf_disease_model_gray.h5")
    st.write("\nModel saved as 'leaf_disease_model_gray.h5'.")

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    st.write(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

    # Visualize epoch-wise accuracy and loss
    st.write("Training Progress:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax1.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax1.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax1.legend()
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    # Loss plot
    ax2.plot(train_losses, label='Train Loss', color='red')
    ax2.plot(val_losses, label='Validation Loss', color='orange')
    ax2.legend()
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    st.pyplot(fig)

    return model

# WebRTC-based webcam class for real-time frame processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = tf.keras.models.load_model('leaf_disease_model_gray.h5')
        self.categories = CATEGORIES

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (256, 256)) / 255.0
        img_array = img_resized.reshape(1, 256, 256, 1)

        predictions = self.model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index] * 100

        # Overlay the predictions on the image
        for i, score in enumerate(predictions):
            label = f"{self.categories[i]}: {score * 100:.2f}%"
            cv2.putText(img, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main Function
def main():
    st.title("Grape Leaf Disease Prediction App")
    st.write("Upload an image or use the webcam to predict the leaf disease.")

    # Upload image option
    file_path, img_array, gray_image = upload_image()

    if img_array is not None:
        st.write("\nLoading pre-trained model (if available)...")
        try:
            model = tf.keras.models.load_model("leaf_disease_model_gray.h5")
            st.write("Model loaded successfully.")
        except:
            st.write("No pre-trained model found. Training a new model...")
            train_dir = r'C:\Users\harsh\Desktop\Grapes final\GRAPES GREY AVERAGE\Grapes Leaves Dataset (images)\train'
            test_dir = r'C:\Users\harsh\Desktop\Grapes final\GRAPES GREY AVERAGE\Grapes Leaves Dataset (images)\test'
            x_train, y_train = preprocess_data(train_dir, CATEGORIES)
            x_test, y_test = preprocess_data(test_dir, CATEGORIES)
            model = train_and_visualize(x_train, y_train, x_test, y_test)

        predict_uploaded_image(model, img_array, file_path, gray_image)

    # Real-time video stream
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
