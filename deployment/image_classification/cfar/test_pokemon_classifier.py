import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import glob

# We will use the same image size as in training
IMG_SIZE = (64, 64)

# Path to the test dataset
TEST_DATASET_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/test"

# List available models in the current directory
def list_available_models():
    models = glob.glob("*.h5")
    return models

def load_test_data():
    X_test, y_test = [], []
    # Load the list of labels from the training directory to ensure same order
    train_dir = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"
    labels = sorted(os.listdir(train_dir))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    print(f"Found {len(labels)} Pokemon classes")
    
    # Load test images
    for label_name in os.listdir(TEST_DATASET_DIR):
        folder = os.path.join(TEST_DATASET_DIR, label_name)
        if not os.path.isdir(folder):
            continue
            
        if label_name not in label_to_idx:
            print(f"Warning: Found test folder '{label_name}' with no matching training class. Skipping.")
            continue
            
        label_idx = label_to_idx[label_name]
        print(f"Loading test images for '{label_name}' (class {label_idx})")
        
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0  # Normalize colors
                X_test.append(img_array)
                y_test.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Loaded {len(X_test)} test images")
    return X_test, y_test, labels

def evaluate_model(model, X_test, y_test, labels):
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=labels))
    

def main():
    # List available models
    available_models = list_available_models()
    
    if not available_models:
        print("Keine Modelle (*.h5) im aktuellen Verzeichnis gefunden!")
        return
    
    # Show available models
    print("Verfügbare Modelle:")
    for i, model_path in enumerate(available_models):
        print(f"{i+1}. {model_path}")
    
    # Ask user to select a model
    while True:
        try:
            selection = input("\nBitte wähle ein Modell (Nummer eingeben oder 'q' zum Beenden): ")
            
            if selection.lower() == 'q':
                print("Programm wird beendet.")
                return
                
            model_index = int(selection) - 1
            if 0 <= model_index < len(available_models):
                selected_model = available_models[model_index]
                break
            else:
                print(f"Ungültige Auswahl. Bitte eine Zahl zwischen 1 und {len(available_models)} eingeben.")
        except ValueError:
            print("Bitte eine gültige Zahl eingeben.")
    
    print(f"Ausgewähltes Modell: {selected_model}")
    
    print("Loading test data...")
    X_test, y_test, labels = load_test_data()
    
    # Convert integer labels to one-hot encoding
    y_test_categorical = to_categorical(y_test, num_classes=len(labels))
    
    print(f"Loading model from {selected_model}...")
    try:
        model = load_model(selected_model)
        print("Model loaded successfully!")
        
        # Show model summary
        model.summary()
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test_categorical, labels)
        
    except Exception as e:
        print(f"Error loading or evaluating model: {e}")

if __name__ == "__main__":
    main()
