import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

model = load_model()

st.title("MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) to get a prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image from bytes
    img = Image.open(io.BytesIO(uploaded_file.getvalue())).convert('L') # Convert to grayscale
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess the image (similar to what you did in the notebook)
    img_array = np.array(img)

    # No inversion needed if digits are white on black, as we fixed in the notebook.
    # If you have black digits on a white background, you might need this:
    # img_array = 255 - img_array

    # Find non-zero pixels (the digit) and crop
    # Use a slightly lower threshold for 'empty' pixels to be more robust
    rows = np.any(img_array > 20, axis=1)
    cols = np.any(img_array > 20, axis=0)

    if np.any(rows) and np.any(cols): # Check if any digit is detected
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cropped = img_array[ymin:ymax+1, xmin:xmax+1]
    else:
        cropped = img_array # Use original if no digit found

    # Resize while maintaining aspect ratio
    cropped_img = Image.fromarray(cropped)
    max_size = 20
    cropped_img.thumbnail((max_size, max_size))

    # Create 28x28 black canvas and center digit
    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - cropped_img.width) // 2
    y_offset = (28 - cropped_img.height) // 2
    canvas.paste(cropped_img, (x_offset, y_offset))

    final_img = np.array(canvas) / 255.0
    final_img = final_img.reshape(1, 28, 28, 1) # Reshape for model input

    st.subheader("Processed Image (for model prediction):")
    st.image(final_img[0, :, :, 0], caption='This is what the model sees.', use_column_width=True)

    # Make prediction
    prediction = model.predict(final_img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Digit: **{predicted_digit}** with **{confidence:.2f}%** confidence!")
