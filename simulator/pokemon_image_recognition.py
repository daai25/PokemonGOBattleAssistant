import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to the trained model
MODEL_PATH = "../deployment/image_classification/CNN/pokemon_cnn_model_64_final.keras"

def load_and_preprocess_image(img_path, target_size=(64, 64)):
    """
    Load and preprocess an image for model prediction
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target size for resizing the image
        
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values

def get_pokemon_classes():
    """
    Get list of Pokémon classes from the dataset directory
    
    Returns:
        list: List of Pokémon class names
    """
    dataset_path = "../data_acquisition/image_dataset/final_pokemon_dataset/test"
    pokemon_classes = sorted([d for d in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, d))])
    return pokemon_classes

def predict_pokemon_from_image(img_path, show_image=False):
    """
    Predict Pokémon name and confidence from an image
    
    Args:
        img_path (str): Path to the image file
        show_image (bool): Whether to display the image
        
    Returns:
        tuple: (pokemon_name, confidence_score)
    """
    # Load the model
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0
    
    # Get Pokémon classes
    pokemon_classes = get_pokemon_classes()
    
    if not pokemon_classes:
        print("Error: Could not retrieve Pokémon classes")
        return None, 0
    
    # Load and preprocess the image
    try:
        processed_img = load_and_preprocess_image(img_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0
    
    # Optional: Show the image
    if show_image:
        plt.figure(figsize=(4, 4))
        img = image.load_img(img_path, target_size=(64, 64))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        plt.show()
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class_index])
    
    # Get the predicted Pokémon name
    if predicted_class_index < len(pokemon_classes):
        pokemon_name = pokemon_classes[predicted_class_index]
    else:
        pokemon_name = "Unknown"
        print(f"Warning: Predicted index {predicted_class_index} is out of range for available classes")
    
    return pokemon_name, confidence

def predict_and_display_results(img_path):
    """
    Predict Pokémon from image and display formatted results
    
    Args:
        img_path (str): Path to the image file
    """
    pokemon_name, confidence = predict_pokemon_from_image(img_path, show_image=True)
    
    if pokemon_name:
        print("\n" + "="*50)
        print(f"PREDICTION RESULTS")
        print("="*50)
        print(f"Pokémon Name: {pokemon_name}")
        print(f"Confidence:   {confidence:.2%}")
        print("="*50 + "\n")
        return pokemon_name, confidence
    else:
        print("Prediction failed.")
        return None, 0

if __name__ == "__main__":
    # Example usage
    # You can replace this with your own image path
    test_image_path = "../data_acquisition/image_dataset/final_pokemon_dataset/test/Pikachu/1.jpg"
    
    if os.path.exists(test_image_path):
        predict_and_display_results(test_image_path)
    else:
        print(f"Test image not found at {test_image_path}")
        print("Please provide a valid image path when using this module.")
