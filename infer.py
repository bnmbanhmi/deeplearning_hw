# infer.py
import tensorflow as tf
import cv2
import numpy as np
from dataset import read_image  # Assuming you have this in dataset.py
import argparse

def load_model(model_path):
    """Loads a trained model."""
    model = tf.keras.models.load_model(model_path)
    return model

def predict_and_save(model, image_path, output_path):
    """Generates a prediction and saves the result."""
    img = read_image(image_path)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    prediction = (prediction > 0.5).astype(np.uint8)  # Thresholding
    prediction = np.squeeze(prediction)  # Remove unnecessary dimensions
    prediction = prediction * 255  # Scale to 0-255 for visibility

    cv2.imwrite(output_path, prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    # Load the model from the provided path
    model_path = 'best_model.h5'  # Update this if your model is saved elsewhere
    model = load_model(model_path)

    # Generate prediction and save the output
    output_path = 'segmented_output.png'  # Output filename
    predict_and_save(model, args.image_path, output_path)