"""
Main script to run the Random Forest model workflow
"""
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Pokemon GO Battle Assistant Random Forest Model")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'],
                        help='Action to perform: train, evaluate, or predict')
    
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.action == 'train':
        print("Training the Random Forest model...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'random_forest_model.py')])
    
    elif args.action == 'evaluate':
        print("Evaluating the Random Forest model...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'evaluate_model.py')])
    
    elif args.action == 'predict':
        print("Running battle prediction...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'predict_battles.py')])

if __name__ == "__main__":
    main()
