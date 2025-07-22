"""
Main script to run the Gradient Boosting model workflow
"""
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Pokemon GO Battle Assistant Gradient Boosting Model")
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'],
                        help='Action to perform: train, evaluate, or predict')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # to run said commands make sure you are in the correct directory then run:
    # python run_model.py train
    if args.action == 'train':
        print("Training the Gradient Boosting model...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'gradient_boosting_model.py')])
    
    elif args.action == 'evaluate':
        print("Evaluating the Gradient Boosting model...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'evaluate_model.py')])
    
    elif args.action == 'predict':
        print("Running battle prediction...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'predict_battles.py')])

if __name__ == "__main__":
    main()
