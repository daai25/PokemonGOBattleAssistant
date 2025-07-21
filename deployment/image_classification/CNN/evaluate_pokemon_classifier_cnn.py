import numpy as np
import os
import matplotlib
# Verwende ein nicht-interaktives Backend, um GUI-Probleme zu vermeiden
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import glob

# Konfiguration
IMG_SIZE = (64, 64)  # Muss mit dem Trainingsmodell übereinstimmen
TEST_DATASET_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/test"

def list_available_models():
    """Listet alle verfügbaren Modelle im aktuellen Verzeichnis auf"""
    models = glob.glob("*.keras") + glob.glob("*.h5")
    return models

def load_test_data():
    """Lädt die Testdaten aus dem angegebenen Verzeichnis"""
    X_test, y_test = [], []
    
    # Lade die Labels aus der gespeicherten Datei, falls vorhanden
    if os.path.exists('pokemon_labels.txt'):
        with open('pokemon_labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"Labels aus Datei geladen: {len(labels)} Klassen")
    else:
        # Alternativ: Lade Labels aus dem Trainingsverzeichnis
        train_dir = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"
        labels = sorted(os.listdir(train_dir))
        print(f"Labels aus Trainingsverzeichnis geladen: {len(labels)} Klassen")
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Lade Testbilder
    print("Lade Testdaten...")
    for label_name in os.listdir(TEST_DATASET_DIR):
        folder = os.path.join(TEST_DATASET_DIR, label_name)
        if not os.path.isdir(folder):
            continue
            
        if label_name not in label_to_idx:
            print(f"Warnung: Testordner '{label_name}' ohne passende Trainingsklasse. Überspringe.")
            continue
            
        label_idx = label_to_idx[label_name]
        print(f"Lade Testbilder für '{label_name}' (Klasse {label_idx})")
        
        images_loaded = 0
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0  # Farben normalisieren
                X_test.append(img_array)
                y_test.append(label_idx)
                images_loaded += 1
            except Exception as e:
                print(f"Fehler beim Laden von {img_path}: {e}")
        
        print(f"  → {images_loaded} Bilder geladen für {label_name}")
    
    # Konvertieren zu NumPy-Arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Testset geladen: {X_test.shape[0]} Bilder, {len(labels)} Klassen")
    return X_test, y_test, labels

def evaluate_model(model, X_test, y_test, labels):
    """Evaluiert das Modell und speichert die Ergebnisse"""
    # One-Hot-Encoding für die Labels
    from tensorflow.keras.utils import to_categorical
    y_test_categorical = to_categorical(y_test, num_classes=len(labels))
    
    # Evaluiere das Modell
    print("Evaluiere Modell...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)
    print(f"Test-Genauigkeit: {test_accuracy:.4f}")
    print(f"Test-Verlust: {test_loss:.4f}")
    
    # Mache Vorhersagen
    print("Berechne Vorhersagen für Konfusionsmatrix...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Erstelle Klassifikationsbericht
    print("\nKlassifikationsbericht:")
    report = classification_report(y_test, y_pred_classes, target_names=labels)
    print(report)
    
    # Speichere Klassifikationsbericht in Datei
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("Klassifikationsbericht gespeichert als 'classification_report.txt'")
    
    # Detaillierte Auswertung pro Pokemon
    print("\n==== DETAILLIERTE AUSWERTUNG PRO POKeMON ====")
    analyze_pokemon_accuracy(X_test, y_test, y_pred, labels)
    
    # Zeige Beispielvorhersagen
    save_example_predictions(X_test, y_test, y_pred, labels, num_examples=10)

def save_example_predictions(X_test, y_test, y_pred, labels, num_examples=10):
    """Speichert einige Beispielvorhersagen als Bild"""
    print(f"Speichere {num_examples} Beispielvorhersagen...")
    # Wähle zufällige Beispiele
    indices = np.random.choice(range(len(X_test)), min(num_examples, len(X_test)), replace=False)
    
    # Erstelle eine Abbildung für die Beispiele
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Hole die tatsächlichen und vorhergesagten Labels
        true_label = labels[y_test[idx]]
        pred_probs = y_pred[idx]
        pred_label = labels[np.argmax(pred_probs)]
        confidence = np.max(pred_probs) * 100
        
        # Zeige das Bild
        axes[i].imshow(X_test[idx])
        correct = true_label == pred_label
        color = "green" if correct else "red"
        
        axes[i].set_title(f"Wahr: {true_label}\nVorhersage: {pred_label}\nSicherheit: {confidence:.1f}%", 
                         color=color, fontsize=10)
        axes[i].axis('off')
        
        # Gib die Vorhersage auch in der Konsole aus
        print(f"Beispiel {i+1}: Wahr: {true_label}, Vorhersage: {pred_label}, " +
              f"Sicherheit: {confidence:.1f}%, {'Korrekt' if correct else 'Falsch'}")
    
    plt.tight_layout()
    plt.savefig('example_predictions.png')
    plt.close()
    print("Beispielvorhersagen gespeichert als 'example_predictions.png'")

def analyze_pokemon_accuracy(X_test, y_test, y_pred, labels):
    """
    Detaillierte Analyse der Genauigkeit und Confidence pro Pokemon
    """
    # Umwandeln der Vorhersagen in Klassenindizes
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Erstellen einer Datei für die detaillierte Auswertung
    with open('pokemon_accuracy_details.txt', 'w') as f:
        f.write("DETAILLIERTE AUSWERTUNG PRO POKEMON\n")
        f.write("===================================\n\n")
        
        # Überschrift für die Konsole
        print(f"{'Pokemon':<20} {'Genauigkeit':<12} {'Durchschn. Confidence':<20} {'Anzahl Testbilder':<15}")
        print("-" * 70)
        
        # Überschrift für die Datei
        f.write(f"{'Pokemon':<20} {'Genauigkeit':<12} {'Durchschn. Confidence':<20} {'Anzahl Testbilder':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Für jede Pokemon-Klasse die Genauigkeit und Confidence berechnen
        for class_idx, pokemon_name in enumerate(labels):
            # Finde alle Testdaten für dieses Pokemon
            class_indices = np.where(y_test == class_idx)[0]
            
            if len(class_indices) == 0:
                # Keine Testdaten für diese Klasse
                accuracy = "N/A"
                avg_confidence = "N/A"
                result = f"{pokemon_name:<20} {accuracy:<12} {avg_confidence:<20} {0:<15}"
                print(result)
                f.write(result + "\n")
                continue
            
            # Berechne die Genauigkeit (korrekte Vorhersagen / alle Vorhersagen)
            correct_predictions = sum(y_pred_classes[class_indices] == class_idx)
            accuracy = correct_predictions / len(class_indices)
            
            # Berechne die durchschnittliche Confidence für korrekte Vorhersagen
            confidences = []
            for idx in class_indices:
                pred_class = y_pred_classes[idx]
                confidence = y_pred[idx][pred_class] * 100  # In Prozent
                confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Ausgabe in Konsole und Datei
            result = f"{pokemon_name:<20} {accuracy:.4f}      {avg_confidence:.2f}%              {len(class_indices):<15}"
            print(result)
            f.write(result + "\n")
            
            # Detaillierte Informationen zu falschen Vorhersagen
            wrong_indices = [idx for idx in class_indices if y_pred_classes[idx] != class_idx]
            if wrong_indices:
                f.write(f"\n  Falsche Vorhersagen für {pokemon_name} ({len(wrong_indices)}/{len(class_indices)}):\n")
                for idx in wrong_indices:
                    pred_class = y_pred_classes[idx]
                    wrong_pokemon = labels[pred_class]
                    confidence = y_pred[idx][pred_class] * 100
                    f.write(f"    - Als {wrong_pokemon} erkannt mit {confidence:.2f}% Confidence\n")
                f.write("\n")
        
        # Gesamtgenauigkeit
        overall_accuracy = np.mean(y_pred_classes == y_test)
        avg_confidence_all = np.mean([np.max(pred) * 100 for pred in y_pred])
        
        summary = f"\nGesamtgenauigkeit: {overall_accuracy:.4f}"
        summary += f"\nDurchschnittliche Confidence: {avg_confidence_all:.2f}%"
        summary += f"\nAnzahl Testbilder gesamt: {len(y_test)}"
        
        print("\n" + summary)
        f.write("\n" + summary)
    
    print(f"\nDetaillierte Auswertung gespeichert als 'pokemon_accuracy_details.txt'")

def main():
    available_models = list_available_models()
    
    if not available_models:
        print("Keine Modelle im aktuellen Verzeichnis gefunden.")
        print("Bitte trainiere zuerst ein Modell oder kopiere ein trainiertes Modell in dieses Verzeichnis.")
        return
    
    # Zeige verfügbare Modelle
    print("Verfügbare Modelle:")
    for i, model_path in enumerate(available_models):
        print(f"{i+1}. {model_path}")
    
    # Nutzer wählt ein Modell aus
    while True:
        try:
            choice = input("\nBitte wähle ein Modell (Nummer) oder 'q' zum Beenden: ")
            
            if choice.lower() == 'q':
                print("Programm wird beendet.")
                return
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                break
            else:
                print(f"Bitte eine Zahl zwischen 1 und {len(available_models)} eingeben.")
        except ValueError:
            print("Bitte eine gültige Zahl eingeben.")
    
    print(f"Ausgewähltes Modell: {selected_model}")
    
    print("Lade Testdaten...")
    X_test, y_test, labels = load_test_data()
    
    print(f"Lade Modell aus {selected_model}...")
    try:
        model = load_model(selected_model)
        print("Modell erfolgreich geladen!")
        
        # Zeige Modellzusammenfassung
        model.summary()
        
        # Evaluiere das Modell
        evaluate_model(model, X_test, y_test, labels)
        
    except Exception as e:
        print(f"Fehler beim Laden oder Evaluieren des Modells: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
