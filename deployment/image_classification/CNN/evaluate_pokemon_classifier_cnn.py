import numpy as np
import os
import matplotlib
# Use a non-interactive backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import glob

# Configuration
IMG_SIZE = (64, 64)  # Must match the training model
TEST_DATASET_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/test"

def list_available_models():
    """Lists all available models in the current directory"""
    models = glob.glob("*.keras") + glob.glob("*.h5")
    return models

def load_test_data():
    """Loads test data from the specified directory"""
    X_test, y_test = [], []
    
    # Load labels from saved file if available
    if os.path.exists('pokemon_labels.txt'):
        with open('pokemon_labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"Labels loaded from file: {len(labels)} classes")
    else:
        # Alternative: Load labels from training directory
        train_dir = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"
        labels = sorted(os.listdir(train_dir))
        print(f"Labels loaded from training directory: {len(labels)} classes")
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Load test images
    print("Loading test data...")
    for label_name in os.listdir(TEST_DATASET_DIR):
        folder = os.path.join(TEST_DATASET_DIR, label_name)
        if not os.path.isdir(folder):
            continue
            
        if label_name not in label_to_idx:
            print(f"Warning: Test folder '{label_name}' has no matching training class. Skipping.")
            continue
            
        label_idx = label_to_idx[label_name]
        print(f"Loading test images for '{label_name}' (Class {label_idx})")
        
        images_loaded = 0
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0  # Normalize colors
                X_test.append(img_array)
                y_test.append(label_idx)
                images_loaded += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"  → {images_loaded} images loaded for {label_name}")
    
    # Convert to NumPy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Test set loaded: {X_test.shape[0]} images, {len(labels)} classes")
    return X_test, y_test, labels

def evaluate_model(model, X_test, y_test, labels):
    """Evaluates the model and saves the results"""
    # One-hot encoding for labels
    from tensorflow.keras.utils import to_categorical
    y_test_categorical = to_categorical(y_test, num_classes=len(labels))
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    print("Calculating predictions for confusion matrix...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred_classes, target_names=labels)
    print(report)
    
    # Save classification report to file
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved as 'classification_report.txt'")
    
    # Detailed evaluation per Pokemon
    print("\n==== DETAILED EVALUATION PER POKEMON ====")
    analyze_pokemon_accuracy(X_test, y_test, y_pred, labels)
    
    # Show example predictions
    save_example_predictions(X_test, y_test, y_pred, labels, num_examples=10)

def save_example_predictions(X_test, y_test, y_pred, labels, num_examples=10):
    """Saves some example predictions as an image"""
    print(f"Saving {num_examples} example predictions...")
    # Choose random examples
    indices = np.random.choice(range(len(X_test)), min(num_examples, len(X_test)), replace=False)
    
    # Create a figure for the examples
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Get actual and predicted labels
        true_label = labels[y_test[idx]]
        pred_probs = y_pred[idx]
        pred_label = labels[np.argmax(pred_probs)]
        confidence = np.max(pred_probs) * 100
        
        # Show the image
        axes[i].imshow(X_test[idx])
        correct = true_label == pred_label
        color = "green" if correct else "red"
        
        axes[i].set_title(f"True: {true_label}\nPrediction: {pred_label}\nConfidence: {confidence:.1f}%", 
                         color=color, fontsize=10)
        axes[i].axis('off')
        
        # Print the prediction in the console
        print(f"Example {i+1}: True: {true_label}, Prediction: {pred_label}, " +
              f"Confidence: {confidence:.1f}%, {'Correct' if correct else 'Incorrect'}")
    
    plt.tight_layout()
    plt.savefig('example_predictions.png')
    plt.close()
    print("Example predictions saved as 'example_predictions.png'")

def analyze_pokemon_accuracy(X_test, y_test, y_pred, labels):
    """
    Detailed analysis of accuracy and confidence per Pokemon
    """
    # Convert predictions to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create a file for detailed evaluation
    with open('pokemon_accuracy_details.txt', 'w') as f:
        f.write("DETAILED EVALUATION PER POKEMON\n")
        f.write("===================================\n\n")
        
        # Header for the console
        print(f"{'Pokemon':<20} {'Accuracy':<12} {'Avg. Confidence':<20} {'Test Images':<15}")
        print("-" * 70)
        
        # Header for the file
        f.write(f"{'Pokemon':<20} {'Accuracy':<12} {'Avg. Confidence':<20} {'Test Images':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Calculate accuracy and confidence for each Pokemon class
        for class_idx, pokemon_name in enumerate(labels):
            # Find all test data for this Pokemon
            class_indices = np.where(y_test == class_idx)[0]
            
            if len(class_indices) == 0:
                # No test data for this class
                accuracy = "N/A"
                avg_confidence = "N/A"
                result = f"{pokemon_name:<20} {accuracy:<12} {avg_confidence:<20} {0:<15}"
                print(result)
                f.write(result + "\n")
                continue
            
            # Calculate accuracy (correct predictions / all predictions)
            correct_predictions = sum(y_pred_classes[class_indices] == class_idx)
            accuracy = correct_predictions / len(class_indices)
            
            # Calculate average confidence for correct predictions
            confidences = []
            for idx in class_indices:
                pred_class = y_pred_classes[idx]
                confidence = y_pred[idx][pred_class] * 100  # In percent
                confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Output to console and file
            result = f"{pokemon_name:<20} {accuracy:.4f}      {avg_confidence:.2f}%              {len(class_indices):<15}"
            print(result)
            f.write(result + "\n")
            
            # Detailed information about incorrect predictions
            wrong_indices = [idx for idx in class_indices if y_pred_classes[idx] != class_idx]
            if wrong_indices:
                f.write(f"\n  Incorrect predictions for {pokemon_name} ({len(wrong_indices)}/{len(class_indices)}):\n")
                for idx in wrong_indices:
                    pred_class = y_pred_classes[idx]
                    wrong_pokemon = labels[pred_class]
                    confidence = y_pred[idx][pred_class] * 100
                    f.write(f"    - Recognized as {wrong_pokemon} with {confidence:.2f}% confidence\n")
                f.write("\n")
        
        # Overall accuracy
        overall_accuracy = np.mean(y_pred_classes == y_test)
        avg_confidence_all = np.mean([np.max(pred) * 100 for pred in y_pred])
        
        summary = f"\nOverall accuracy: {overall_accuracy:.4f}"
        summary += f"\nAverage confidence: {avg_confidence_all:.2f}%"
        summary += f"\nTotal test images: {len(y_test)}"
        
        print("\n" + summary)
        f.write("\n" + summary)
    
    print(f"\nDetailed evaluation saved as 'pokemon_accuracy_details.txt'")

def main():
    available_models = list_available_models()
    
    if not available_models:
        print("No models found in the current directory.")
        print("Please train a model first or copy a trained model into this directory.")
        return
    
    # Show available models
    print("Available models:")
    for i, model_path in enumerate(available_models):
        print(f"{i+1}. {model_path}")
    
    # User selects a model
    while True:
        try:
            choice = input("\nPlease choose a model (number) or 'q' to quit: ")
            
            if choice.lower() == 'q':
                print("Program terminated.")
                return
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_models)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"Selected model: {selected_model}")
    
    print("Loading test data...")
    X_test, y_test, labels = load_test_data()
    
    print(f"Loading model from {selected_model}...")
    try:
        model = load_model(selected_model)
        print("Model loaded successfully!")
        
        # Show model summary
        model.summary()
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test, labels)
        
    except Exception as e:
        print(f"Error loading or evaluating the model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
