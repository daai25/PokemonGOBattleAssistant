import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import sys

class PokemonClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokemon Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.model = None
        self.labels = []
        self.img_size = (64, 64)  # Default image size, must match the model
        
        self.create_ui()
    
    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Pokémon GO Battle Assistant", 
            font=("Arial", 24, "bold"),
            bg="#f0f0f0",
            fg="#1a1a1a"
        )
        title_label.pack(pady=(0, 20))
        
        # Model selection frame
        model_frame = tk.LabelFrame(
            main_frame, 
            text="Select Model", 
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#1a1a1a",
            padx=10,
            pady=10
        )
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection button
        self.model_path_var = tk.StringVar()
        self.model_path_entry = tk.Entry(
            model_frame, 
            textvariable=self.model_path_var,
            width=50,
            font=("Arial", 10)
        )
        self.model_path_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        model_select_btn = tk.Button(
            model_frame, 
            text="Choose Model", 
            command=self.select_model,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10
        )
        model_select_btn.pack(side=tk.RIGHT)
        
        # Image frame
        image_frame = tk.LabelFrame(
            main_frame, 
            text="Upload Image", 
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#1a1a1a",
            padx=10,
            pady=10
        )
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Image display
        self.image_label = tk.Label(
            image_frame,
            text="No image selected",
            bg="#e0e0e0",
            width=60,
            height=15
        )
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Image selection button
        image_select_btn = tk.Button(
            image_frame, 
            text="Select Image", 
            command=self.select_image,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10
        )
        image_select_btn.pack(side=tk.LEFT, padx=10, pady=(0, 10))
        
        # Classify button
        self.classify_btn = tk.Button(
            image_frame, 
            text="Classify", 
            command=self.classify_image,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            state=tk.DISABLED
        )
        self.classify_btn.pack(side=tk.RIGHT, padx=10, pady=(0, 10))
        
        # Result frame
        result_frame = tk.LabelFrame(
            main_frame, 
            text="Result", 
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#1a1a1a",
            padx=10,
            pady=10
        )
        result_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Result label
        self.result_var = tk.StringVar(value="No classification performed yet")
        result_label = tk.Label(
            result_frame, 
            textvariable=self.result_var,
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="#1a1a1a",
            padx=10,
            pady=10
        )
        result_label.pack(fill=tk.X)
        
        # Top-3 results
        self.top3_frame = tk.Frame(result_frame, bg="#f0f0f0")
        self.top3_frame.pack(fill=tk.X, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9),
            bg="#e0e0e0",
            fg="#1a1a1a"
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[
                ("Keras Models", "*.keras"),
                ("HDF5 Models", "*.h5"),
                ("All Files", "*.*")
            ]
        )
        
        if not model_path:
            return
        
        self.model_path_var.set(model_path)
        self.status_var.set("Loading model...")
        self.root.update()
        
        try:
            self.model = load_model(model_path)
            
            # Load the labels
            model_dir = os.path.dirname(model_path)
            labels_path = os.path.join(model_dir, "pokemon_labels.txt")
            
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
            else:
                # Search for labels file in current directory
                if os.path.exists("pokemon_labels.txt"):
                    with open("pokemon_labels.txt", 'r') as f:
                        self.labels = [line.strip() for line in f.readlines()]
                else:
                    messagebox.showwarning(
                        "Warning", 
                        "No pokemon_labels.txt file found. Class names will be displayed numerically."
                    )
            
            self.status_var.set(f"Model loaded: {os.path.basename(model_path)} with {len(self.labels)} classes")
            self.classify_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.status_var.set("Error loading model")
            self.model = None
            self.labels = []
            self.classify_btn.config(state=tk.DISABLED)
    
    def select_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png"),
                ("All Files", "*.*")
            ]
        )
        
        if not image_path:
            return
        
        self.status_var.set(f"Image selected: {os.path.basename(image_path)}")
        self.image_path = image_path
        
        # Show the image
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Resize for display, not for classification
            display_size = (300, 300)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update the label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Enable classify button if model is loaded
            if self.model is not None:
                self.classify_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            self.status_var.set("Error loading image")
            self.image_label.config(image=None, text="Error loading image")
    
    def classify_image(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
        
        if not hasattr(self, 'image_path'):
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        self.status_var.set("Classifying image...")
        self.root.update()
        
        try:
            # Load and prepare the image
            img = load_img(self.image_path, target_size=self.img_size)
            img_array = img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Prediction
            predictions = self.model.predict(img_array)[0]
            
            # Find top-3 predictions
            top3_indices = predictions.argsort()[-3:][::-1]
            top3_scores = predictions[top3_indices]
            
            # Prepare class names
            if self.labels:
                if len(self.labels) > max(top3_indices):
                    top3_classes = [self.labels[i] for i in top3_indices]
                else:
                    top3_classes = [f"Class {i}" for i in top3_indices]
            else:
                top3_classes = [f"Class {i}" for i in top3_indices]
            
            # Show the result
            self.result_var.set(f"This Pokémon is most likely: {top3_classes[0]} ({top3_scores[0]*100:.1f}%)")
            
            # Delete old widgets in the top3 frame
            for widget in self.top3_frame.winfo_children():
                widget.destroy()
            
            # Show top-3 results
            for i in range(len(top3_classes)):
                result_text = f"{i+1}. {top3_classes[i]}: {top3_scores[i]*100:.1f}%"
                
                # Color code based on probability
                if top3_scores[i] > 0.7:
                    bg_color = "#4CAF50"  # Green for high probability
                elif top3_scores[i] > 0.4:
                    bg_color = "#FF9800"  # Orange for medium probability
                else:
                    bg_color = "#F44336"  # Red for low probability
                
                result_label = tk.Label(
                    self.top3_frame,
                    text=result_text,
                    font=("Arial", 12, "bold" if i == 0 else "normal"),
                    bg=bg_color,
                    fg="white",
                    padx=10,
                    pady=5
                )
                result_label.pack(fill=tk.X, pady=2)
            
            self.status_var.set("Classification complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during classification: {str(e)}")
            self.status_var.set("Error during classification")

def main():
    root = tk.Tk()
    app = PokemonClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
