# test.py
import tensorflow as tf
import cv2
import numpy as np
from dataset import read_image
from config import SAVE_WEIGHTS_PATH, TEST_DATA_PATH, SAVE_RESULTS_PATH, IMG_HEIGHT, IMG_WIDTH

def load_model(model_path):
    """Loads a trained model."""
    model = tf.keras.models.load_model(model_path)
    return model

def predict_and_save(model, image_path, output_path):
    """Generates prediction and saves the result."""
    img = read_image(image_path)  # Use the read_image function from dataset.py
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    prediction = (prediction > 0.5).astype(np.uint8)  # Thresholding
    prediction = np.squeeze(prediction, axis=0)  # Remove batch dimension
    prediction = cv2.resize(prediction, (IMG_WIDTH, IMG_HEIGHT))  # Resize to original dimensions
    prediction = prediction * 255  # Scale to 0-255

    cv2.imwrite(output_path, prediction)

if __name__ == "__main__":
    # Load the trained model
    model_path = os.path.join(SAVE_WEIGHTS_PATH, 'trained_model.h5')  # Or 'best_model.h5'
    model = load_model(model_path)

    # Load and predict on test images
    test_image_paths = [os.path.join(TEST_DATA_PATH, f) for f in os.listdir(TEST_DATA_PATH) if f.endswith('.png')]
    for image_path in test_image_paths:
        filename = os.path.basename(image_path)
        output_path = os.path.join(SAVE_RESULTS_PATH, filename)
        predict_and_save(model, image_path, output_path)