from PIL import Image
import numpy as np
import tensorflow as tf

def is_rgb_image(img):
    return img.mode == 'RGB'

def convert_to_rgb(img):
    return img.convert('RGB')

def predict_single_image(img_file, model, img_size=(299, 299)):
    try:
        # Load and preprocess the image
        img = Image.open(img_file)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize(img_size)  # Resize image
        img_array = np.array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Validate image shape
        expected_shape = (1,) + img_size + (3,)  # Assuming 3 channels for RGB images
        if img_array.shape != expected_shape:
            raise ValueError(f"Invalid image shape. Expected {expected_shape}, got {img_array.shape}.")

        # Predict with the model
        predictions = model.predict(img_array)
        return predictions

    except Exception as e:
        print(f"Error predicting image: {e}")
        return None
