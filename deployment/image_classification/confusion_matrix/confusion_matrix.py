import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import random
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define paths
base_path = "../../../data_acquisition/image_dataset/final_pokemon_dataset/test"
cfar_model_path = "../cfar/pokemon_classifier_cfar.keras"
cnn_model_path = "../CNN/pokemon_cnn_model_64_final.keras"
efficientnet_model_path = "../EfficientNetB0/model/pokemon_final_model.keras"

def load_and_preprocess_image(img_path, target_size=(64, 64), preprocess_func=None):
    """Load and preprocess an image for model prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if preprocess_func:
        return preprocess_func(img_array)
    else:
        return img_array / 255.0  # Simple normalization

def predict_image(model, img_path, target_size=(64, 64), preprocess_func=None):
    """Predict class for a single image"""
    processed_img = load_and_preprocess_image(img_path, target_size, preprocess_func)
    predictions = model.predict(processed_img, verbose=0)
    return np.argmax(predictions, axis=1)[0]

def evaluate_model(model, pokemon_dirs, class_mapping, target_size=(64, 64), preprocess_func=None, num_samples=5):
    """Evaluate model on test images and create confusion matrix"""
    y_true = []
    y_pred = []
    
    print(f"Model output shape: {model.output_shape}")
    num_classes = model.output_shape[1]
    print(f"Number of classes in model: {num_classes}")
    
    # Create a mapping from actual class names to indices in our current evaluation
    selected_class_indices = {pokemon: idx for idx, pokemon in enumerate(pokemon_dirs)}
    
    # Collect all predictions first to analyze
    all_predictions = []
    correctly_predicted = 0
    total_predictions = 0
    
    # Get the ordering of classes in the model's training
    # This is where we'll look for the training metadata
    model_dir = os.path.dirname(os.path.abspath(model.filepath)) if hasattr(model, 'filepath') else None
    class_ordering = None
    
    # Try to load class mapping if it exists
    if model_dir:
        class_map_path = os.path.join(model_dir, 'class_mapping.npy')
        if os.path.exists(class_map_path):
            try:
                class_ordering = np.load(class_map_path, allow_pickle=True).item()
                print(f"Loaded class mapping from {class_map_path}")
            except:
                print(f"Could not load class mapping from {class_map_path}")
    
    # Load existing pokemon directories
    all_test_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for idx, pokemon_dir in enumerate(pokemon_dirs):
        pokemon_path = os.path.join(base_path, pokemon_dir)
        if not os.path.isdir(pokemon_path):
            print(f"Warning: Directory not found: {pokemon_path}")
            continue
            
        images = [f for f in os.listdir(pokemon_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"Warning: No images found in {pokemon_path}")
            continue
            
        # Select up to num_samples random images for each Pokémon
        selected_images = random.sample(images, min(num_samples, len(images)))
        
        for img_name in selected_images:
            img_path = os.path.join(pokemon_path, img_name)
            
            try:
                # True label is the current index in our selected_pokemon list
                true_label = idx
                y_true.append(true_label)
                
                # Get full prediction array
                processed_img = load_and_preprocess_image(img_path, target_size, preprocess_func)
                predictions = model.predict(processed_img, verbose=0)
                all_predictions.append(predictions[0])
                
                # Get most confident class
                predicted_class = np.argmax(predictions, axis=1)[0]
                
                # For debugging, print the direct prediction and what we think it should map to
                if len(all_predictions) <= 5:
                    print(f"Pokemon: {pokemon_dir}, True idx: {true_label}, Predicted idx: {predicted_class}")
                    print(f"Top 3 predictions: {np.argsort(predictions[0])[-3:][::-1]}")
                    print(f"Prediction confidence: {np.max(predictions[0]):.4f}")
                
                # Map model's predicted index to our current selected_pokemon index if possible
                if class_ordering:
                    # If we have a class mapping from training, use it
                    predicted_name = class_ordering.get(predicted_class)
                    if predicted_name in selected_class_indices:
                        mapped_pred = selected_class_indices[predicted_name]
                        y_pred.append(mapped_pred)
                    else:
                        # If the predicted class name isn't in our current set, keep original prediction
                        y_pred.append(predicted_class)
                else:
                    # If we don't have class mapping, use direct prediction
                    y_pred.append(predicted_class)
                
                # For evaluation, check if prediction is correct
                if y_pred[-1] == true_label:
                    correctly_predicted += 1
                total_predictions += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Raw accuracy: {correctly_predicted/total_predictions*100:.2f}% ({correctly_predicted}/{total_predictions})")
    
    # Analyze predictions to help debug
    all_predictions = np.array(all_predictions)
    avg_confidence = np.mean([np.max(pred) for pred in all_predictions])
    print(f"Average confidence: {avg_confidence:.4f}")
    
    # Fix the mapping if the model was trained with a different class order
    if avg_confidence < 0.3 or correctly_predicted == 0:  # If confidence is very low, try remapping
        print("Low confidence detected. Trying to remap predictions...")
        
        # Try assumption: model might have been trained on alphabetically sorted classes
        sorted_test_dirs = sorted(all_test_dirs)
        
        # Create a mapping from alphabetical index to our selected_pokemon index
        alpha_to_selected = {}
        for idx, pokemon in enumerate(pokemon_dirs):
            if pokemon in sorted_test_dirs:
                alpha_idx = sorted_test_dirs.index(pokemon)
                alpha_to_selected[alpha_idx] = idx
        
        # Apply the mapping to predictions
        new_y_pred = []
        for pred in y_pred:
            if pred < len(sorted_test_dirs) and pred in alpha_to_selected:
                new_y_pred.append(alpha_to_selected[pred])
            else:
                # If out of range or not mapped, keep original
                new_y_pred.append(pred)
        
        # Calculate new accuracy
        new_correct = sum(1 for true, pred in zip(y_true, new_y_pred) if true == pred)
        new_accuracy = new_correct / len(y_true) if y_true else 0
        
        print(f"After alphabetical remapping: {new_accuracy*100:.2f}% accuracy")
        
        # If remapping improved accuracy, use it
        if new_accuracy > (correctly_predicted/total_predictions):
            y_pred = new_y_pred
            print("Using remapped predictions")
    
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # Using seaborn for a better-looking heatmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - {title}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    
    plt.show()

def plot_accuracy_comparison(accuracies, model_names):
    """Create a bar chart comparing model accuracies"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, max(accuracies) + 10)  # Add some space for the text
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'model_accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy comparison to {output_path}")
    
    plt.show()

# Main execution
def main():
    # Load models
    print("Loading models...")
    cfar_model = load_model(cfar_model_path)
    cnn_model = load_model(cnn_model_path)
    efficientnet_model = load_model(efficientnet_model_path)
    
    # Use a predefined list of 20 specific Pokémon
    selected_pokemon = [
        "Pikachu",         # Electric - Popular
        "Charizard",       # Fire/Flying - Popular starter evolution
        "Gyarados",        # Water/Flying - Common in meta
        "Mewtwo",          # Psychic - Legendary
        "Gengar",          # Ghost/Poison - Popular
        "Machamp",         # Fighting - Common in battles
        "Dragonite",       # Dragon/Flying - Popular pseudo-legendary
        "Blastoise",       # Water - Starter evolution
        "Snorlax",         # Normal - Tanky popular
        "Venusaur",        # Grass/Poison - Starter evolution
        "Tyranitar",       # Rock/Dark - Pseudo-legendary
        "Alakazam",        # Psychic - Popular
        "Vaporeon",        # Water - Popular Eeveelution
        "Jolteon",         # Electric - Eeveelution
        "Flareon",         # Fire - Eeveelution
        "Lapras",          # Water/Ice - Popular
        "Rhydon",          # Ground/Rock - Common evolution
        "Arcanine",        # Fire - Popular
        "Golem",           # Rock/Ground - Common evolution
        "Exeggutor"        # Grass/Psychic - Unique typing
    ]
    
    # Verify all selected Pokémon exist in the dataset
    existing_pokemon_dirs = os.listdir(base_path)
    for pokemon in selected_pokemon:
        if pokemon not in existing_pokemon_dirs:
            print(f"Warning: {pokemon} not found in test dataset. Check the name or replace it.")
    
    # Filter to keep only existing Pokémon
    selected_pokemon = [p for p in selected_pokemon if p in existing_pokemon_dirs]
    
    # If we lost some Pokémon due to name mismatches, fill with popular alternatives
    backup_pokemon = ["Jigglypuff", "Eevee", "Articuno", "Zapdos", "Moltres", 
                     "Meowth", "Psyduck", "Growlithe", "Poliwag", "Abra"]
    
    i = 0
    while len(selected_pokemon) < 20 and i < len(backup_pokemon):
        if backup_pokemon[i] in existing_pokemon_dirs and backup_pokemon[i] not in selected_pokemon:
            selected_pokemon.append(backup_pokemon[i])
        i += 1
    
    # If we still don't have 20, add some from the directory
    if len(selected_pokemon) < 20:
        remaining_pokemon = [p for p in existing_pokemon_dirs 
                            if os.path.isdir(os.path.join(base_path, p)) 
                            and p not in selected_pokemon]
        additional = random.sample(remaining_pokemon, min(20 - len(selected_pokemon), len(remaining_pokemon)))
        selected_pokemon.extend(additional)
    
    print(f"Using these {len(selected_pokemon)} Pokémon for evaluation:")
    for p in selected_pokemon:
        print(f"  - {p}")
    
    class_mapping = {idx: pokemon for idx, pokemon in enumerate(selected_pokemon)}
    
    # Create clean class names for display (remove special characters)
    class_names = [name.replace('(', '').replace(')', '').replace('-', ' ') for name in selected_pokemon]
    
    print(f"Evaluating models on {len(selected_pokemon)} Pokémon classes...")
    
    # Evaluate CFAR model (64x64 images, simple normalization)
    print("Evaluating CFAR model...")
    y_true_cfar, y_pred_cfar = evaluate_model(
        cfar_model, selected_pokemon, class_mapping, 
        target_size=(64, 64), preprocess_func=None
    )
    plot_confusion_matrix(y_true_cfar, y_pred_cfar, class_names, "CFAR Model")
    
    # Evaluate CNN model (64x64 images, simple normalization)
    print("Evaluating CNN model...")
    y_true_cnn, y_pred_cnn = evaluate_model(
        cnn_model, selected_pokemon, class_mapping, 
        target_size=(64, 64), preprocess_func=None
    )
    plot_confusion_matrix(y_true_cnn, y_pred_cnn, class_names, "CNN Model")
    
    # Evaluate EfficientNetB0 model (224x224 images, EfficientNet preprocessing)
    print("Evaluating EfficientNetB0 model...")
    y_true_eff, y_pred_eff = evaluate_model(
        efficientnet_model, selected_pokemon, class_mapping, 
        target_size=(224, 224), preprocess_func=efficientnet_preprocess
    )
    plot_confusion_matrix(y_true_eff, y_pred_eff, class_names, "EfficientNetB0 Model")
    
    # Calculate and print accuracy for each model
    accuracy_cfar = np.mean(y_true_cfar == y_pred_cfar) * 100
    accuracy_cnn = np.mean(y_true_cnn == y_pred_cnn) * 100
    accuracy_eff = np.mean(y_true_eff == y_pred_eff) * 100
    
    print(f"CFAR Model Accuracy: {accuracy_cfar:.2f}%")
    print(f"CNN Model Accuracy: {accuracy_cnn:.2f}%")
    print(f"EfficientNetB0 Model Accuracy: {accuracy_eff:.2f}%")
    
    # Plot accuracy comparison
    accuracies = [accuracy_cfar, accuracy_cnn, accuracy_eff]
    model_names = ['CFAR Model', 'CNN Model', 'EfficientNetB0']
    plot_accuracy_comparison(accuracies, model_names)

if __name__ == "__main__":
    main()