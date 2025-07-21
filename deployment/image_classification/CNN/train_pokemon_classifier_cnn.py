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

# Setze Seed für Reproduzierbarkeit
np.random.seed(42)
tf.random.set_seed(42)

# Konfiguration
IMG_SIZE = (64, 64)  # Höhere Auflösung für bessere Merkmalsextraktion
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001 
DATASET_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"

def load_dataset():
    print("Lade Bilder...")
    X, y = [], []
    # Bilder werden durch das Verzeichnis, in dem sie sich befinden, gekennzeichnet
    labels = sorted(os.listdir(DATASET_DIR))
    print(f"Gefundene Klassen: {len(labels)}")
    
    for label_idx, label_name in enumerate(labels):
        folder = os.path.join(DATASET_DIR, label_name)
        if not os.path.isdir(folder):
            continue
        
        print(f"Lade Klasse {label_idx+1}/{len(labels)}: {label_name}")
        images_loaded = 0
        
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                # Farben normalisieren
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(label_idx)
                images_loaded += 1
            except Exception as e:
                print(f"Fehler beim Laden von {img_path}: {e}")
        
        print(f"  → {images_loaded} Bilder geladen für {label_name}")
    
    # Konvertieren zu NumPy-Arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset geladen: {X.shape[0]} Bilder, {len(labels)} Klassen")
    return X, y, labels

def create_cnn_model(input_shape, num_classes):
    """Erstellt ein verbessertes CNN-Modell für Pokémon-Klassifikation"""
    model = Sequential([
        # Erster Konvolutionsblock
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Zweiter Konvolutionsblock
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dritter Konvolutionsblock
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Klassifikationsteil
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Kompilieren des Modells mit optimiertem Optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_and_save_training_history(history):
    """Visualisiert und speichert die Trainingshistorie"""
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Modell-Genauigkeit')
    plt.ylabel('Genauigkeit')
    plt.xlabel('Epoche')
    plt.legend(loc='lower right')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Modell-Verlust')
    plt.ylabel('Verlust')
    plt.xlabel('Epoche')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Trainingshistorie gespeichert als 'training_history.png'")
    plt.show()

def main():
    # Daten laden
    X, y, labels = load_dataset()
    
    # Wenn keine Bilder geladen wurden, beenden
    if len(X) == 0:
        print("Keine Bilder gefunden. Bitte überprüfe den Pfad zum Datensatz.")
        return
    
    # Labels in One-Hot-Encoding konvertieren
    y_categorical = to_categorical(y, num_classes=len(labels))
    
    # Train-Validation Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Trainingsset: {X_train.shape[0]} Bilder")
    print(f"Validierungsset: {X_val.shape[0]} Bilder")
    
    # Datenaugmentierung
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Datengenerator auf Trainingsdaten anwenden
    datagen.fit(X_train)
    
    # CNN-Modell erstellen
    model = create_cnn_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=len(labels)
    )
    
    # Modellzusammenfassung anzeigen
    model.summary()
    
    # Callbacks für verbessertes Training
    callbacks = [
        # Früher Stopp, wenn sich die Validation Loss nicht verbessert
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Lernrate reduzieren, wenn Plateau erreicht
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Modell trainieren
    print("Starte Training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Finale Evaluierung
    print("Evaluiere finales Modell...")
    final_val_loss, final_val_acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Finale Validierungsgenauigkeit: {final_val_acc:.4f}")
    print(f"Finaler Validierungsverlust: {final_val_loss:.4f}")
    
    # Speichere Labels für die Vorhersage
    with open('pokemon_labels.txt', 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print("Labels gespeichert als 'pokemon_labels.txt'")
    
    # Speichere das Modell
    try:
        model_name = f"pokemon_cnn_model_{int(final_val_acc*100)}.keras"
        model.save(model_name)
        print(f"Modell gespeichert als '{model_name}'")
    except Exception as e:
        print(f"Fehler beim Speichern im .keras-Format: {e}")
        try:
            model_name = f"pokemon_cnn_model_{int(final_val_acc*100)}.h5"
            model.save(model_name)
            print(f"Modell gespeichert als '{model_name}'")
        except Exception as e:
            print(f"Fehler beim Speichern im .h5-Format: {e}")
            # Letzter Versuch mit TensorFlow SavedModel Format
            model_name = f"pokemon_cnn_model_{int(final_val_acc*100)}"
            save_model(model, model_name)
            print(f"Modell gespeichert als '{model_name}' (SavedModel Format)")
    
    # Zeige und speichere Trainingshistorie
    plot_and_save_training_history(history)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
