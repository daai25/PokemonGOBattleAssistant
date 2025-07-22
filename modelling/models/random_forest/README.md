# Pokemon GO Battle Assistant - Random Forest Model

This directory contains scripts for training, evaluating, and making predictions with a Random Forest model for Pokemon GO battles.

## Files Description

- `random_forest_model.py`: Script to train the Random Forest model
- `predict_battles.py`: Script to make battle predictions using the trained model
- `evaluate_model.py`: Script to evaluate the model's performance
- `run_model.py`: Main script to run the different components

## Data Used

The model uses the following data sources:

1. `all_overall_rankings_vectorized.csv`: Contains Pokemon properties and rankings
2. `battle_data_numeric.csv`: Contains battle outcomes and Pokemon stats during battles

## How to Use

### Training the Model

To train the Random Forest model:

```
python run_model.py train
```

This will:
- Load the Pokemon and battle data
- Prepare the features for training
- Train a Random Forest model with hyperparameter tuning
- Save the trained model and scaler to the current directory

### Evaluating the Model

To evaluate the model's performance:

```
python run_model.py evaluate
```

This will:
- Load the trained model and battle data
- Generate performance metrics (accuracy, precision, recall, F1 score)
- Create visualizations (confusion matrix, ROC curve, feature importance)
- Save evaluation results to the `evaluation` folder

### Making Predictions

To predict battle outcomes:

```
python run_model.py predict
```

This will:
- Load the trained model and Pokemon data
- Allow you to input two Pokemon names
- Predict the winner and display the confidence level

## Model Features

The model uses the following features to predict battle outcomes:

### Pokemon Features
- Type 1 and Type 2
- Fast Move Type
- Charged Move Types
- Attack, Defense, and Stamina stats
- Overall score/ranking

## Requirements

Required Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Note

Make sure to train the model before attempting to evaluate or make predictions with it.
