"""
Main script to run the CatBoost model workflow
"""
import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Pokemon GO Battle Assistant CatBoost Model"
    )
    parser.add_argument(
        'action', choices=['train', 'evaluate', 'predict'],
        help='Action to perform: train, evaluate, or predict'
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.action == 'train':
        print("Training the CatBoost model...")
        subprocess.run([
            sys.executable,
            os.path.join(current_dir, 'cat_boost_model.py')
        ])
    elif args.action == 'evaluate':
        print("Evaluating the CatBoost model...")
        subprocess.run([
            sys.executable,
            os.path.join(current_dir, 'evaluate_model.py')
        ])
    elif args.action == 'predict':
        print("Running battle prediction...")
        subprocess.run([
            sys.executable,
            os.path.join(current_dir, 'predict_battles.py')
        ])


if __name__ == '__main__':
    main()