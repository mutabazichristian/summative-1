# Linear Regression Analysis Assignment

## Overview
This assignment explores different approaches to implementing linear regression for predicting sales based on TV marketing expenses. We compare three different methods and analyze their performance:
1. NumPy implementation
2. Scikit-learn implementation
3. Gradient Descent implementation from scratch

Additionally, we compare Linear Regression with Random Forest and Decision Tree algorithms.

## Dataset
- File: `data/tvmarketing.csv`
- Features: TV marketing expenses and corresponding sales data
- Simple dataset with two columns: 'TV' and 'Sales'

## Implementation Details

### 1. NumPy Approach
- Uses `np.polyfit()` for fitting the linear regression line
- Implements basic prediction functionality
- Demonstrates fundamental linear regression concepts

### 2. Scikit-learn Approach
- Uses `sklearn.linear_model.LinearRegression`
- Includes train-test split (80-20)
- Calculates RMSE for model evaluation
- Compares performance with other algorithms:
  - Random Forest
  - Decision Trees

### 3. Gradient Descent Implementation
- Custom implementation from scratch
- Includes:
  - Cost function implementation
  - Partial derivatives calculation
  - Gradient descent optimization
  - Data normalization
  - Result denormalization for predictions

## Requirements
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
```

## Key Learning Outcomes
- Understanding different approaches to linear regression
- Working with data preprocessing (normalization)
- Implementing gradient descent from scratch
- Comparing different machine learning algorithms
- Using various Python libraries for data analysis

## Notes
- The gradient descent implementation requires careful tuning of learning rate and iterations
- Data normalization is crucial for efficient gradient descent convergence
- Model performance is evaluated using RMSE (Root Mean Square Error)
- Includes visualization of results using matplotlib

## Files Structure
```
.
├── data/
│   └── tvmarketing.csv
├── w2_unittest.py
└── main notebook
```

## How to Run
1. Load the dataset using pandas
2. Run each implementation section separately
3. Compare results between different approaches
4. Analyze model performance using RMSE metrics
5. Visualize results using provided plotting functions

## Results
The assignment compares three different regression models:
- Linear Regression
- Random Forest
- Decision Trees

Results show varying RMSE values for each model, allowing for performance comparison.
