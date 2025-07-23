import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (64, 64)  # Higher resolution for better feature extraction
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001 
TRAIN_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"
TEST_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/test"

def load_dataset(directory):
    print(f"Loading images from {directory}...")
    X, y = [], []
    # Images are labeled by the directory they are in
    labels = sorted(os.listdir(directory))
    print(f"Found classes: {len(labels)}")
    
    for label_idx, label_name in enumerate(labels):
        folder = os.path.join(directory, label_name)
        if not os.path.isdir(folder):
            continue
        
        print(f"Loading class {label_idx+1}/{len(labels)}: {label_name}")
        images_loaded = 0
        
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                # Normalize colors
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(label_idx)
                images_loaded += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"  → {images_loaded} images loaded for {label_name}")
    
    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {X.shape[0]} images, {len(labels)} classes")
    return X, y, labels

def create_cnn_model(input_shape, num_classes):
    """Creates an improved CNN model for Pokémon classification"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Classification part
        Flatten(),
        Dense(512, activation='relu'),  # Larger dense layer for more features
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with optimized optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_and_save_training_history(history):
    """Visualizes and saves the training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Test')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")
    plt.show()

def main():
    # Load data
    X_train, y_train, train_labels = load_dataset(TRAIN_DIR)
    X_test, y_test, test_labels = load_dataset(TEST_DIR)
    
    # Check if labels in both datasets match
    if train_labels != test_labels:
        print("WARNING: The labels in the training and test datasets do not match!")
        # Use the labels from the training dataset
        labels = train_labels
    else:
        labels = train_labels
    
    # If no images were loaded, exit
    if len(X_train) == 0 or len(X_test) == 0:
        print("No sufficient images found. Please check the paths to the dataset.")
        return
    
    # Convert labels to one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes=len(labels))
    y_test_categorical = to_categorical(y_test, num_classes=len(labels))
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Apply data generator to training data
    datagen.fit(X_train)
    
    # Create CNN model
    model = create_cnn_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=len(labels)
    )
    
    # Display model summary
    model.summary()
    
    # Callbacks for improved training
    callbacks = [
        # Early stopping if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        datagen.flow(X_train, y_train_categorical, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test_categorical),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("Evaluating final model...")
    final_test_loss, final_test_acc = model.evaluate(X_test, y_test_categorical, verbose=1)
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")
    
    # Save labels for prediction
    with open('pokemon_labels.txt', 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print("Labels saved as 'pokemon_labels.txt'")
    
    # Save the model
    try:
        model_name = f"pokemon_cnn_model_{int(final_test_acc*100)}.keras"
        model.save(model_name)
        print(f"Model saved as '{model_name}'")
    except Exception as e:
        print(f"Error saving in .keras format: {e}")
        try:
            model_name = f"pokemon_cnn_model_{int(final_test_acc*100)}.h5"
            model.save(model_name)
            print(f"Model saved as '{model_name}'")
        except Exception as e:
            print(f"Error saving in .h5 format: {e}")
            # Last attempt with TensorFlow SavedModel format
            model_name = f"pokemon_cnn_model_{int(final_test_acc*100)}"
            save_model(model, model_name)
            print(f"Model saved as '{model_name}' (SavedModel format)")
    
    # Show and save training history
    plot_and_save_training_history(history)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
