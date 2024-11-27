# dataset.py
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import IMG_HEIGHT, IMG_WIDTH

def load_data(images_path, masks_path):
    """Loads image and mask paths."""
    image_paths = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.png')]
    mask_paths = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png')]
    return image_paths, mask_paths

def read_image(image_path):
    """Reads and preprocesses an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0  # Normalize
    return image

def read_mask(mask_path):
    """Reads and preprocesses a mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    mask = mask / 255.0  # Normalize
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
    return mask

def show_example(image, mask):
    """Displays an image and its corresponding mask."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')  # Show the first channel of the mask
    plt.title('Mask')
    plt.show()

def convert2TfDataset(image_paths, mask_paths, batch_size):
    """Creates a TensorFlow Dataset."""
    def _load_image_and_mask(image_path, mask_path):
        image = tf.numpy_function(read_image, [image_path], tf.float32)
        mask = tf.numpy_function(read_mask, [mask_path], tf.float32)
        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(_load_image_and_mask)
    dataset = dataset.batch(batch_size)
    return dataset