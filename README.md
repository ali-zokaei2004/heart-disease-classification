# Heart Disease Classification Project

## Project Overview
This project implements a comprehensive machine learning workflow for heart disease classification using multiple algorithms. The code performs data preprocessing, model training, evaluation, and visualization to compare the performance of different classification models.

## Dataset
- **Source**: heart.csv
- **Target Variable**: "target" (indicating presence/absence of heart disease)
- **Features**: Various medical parameters

## Requirements
### Python Libraries
- pandas
- matplotlib
- seaborn
- scikit-learn

### Installation
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Methodology

### 1. Data Preprocessing
- Loaded dataset from CSV file
- Split data into training (80%) and testing (20%) sets with stratification
- Applied Min-Max scaling to normalize features between 0 and 1

### 2. Models Implemented
The project evaluates five classification algorithms:
1. **Naive Bayes** (GaussianNB)
2. **K-Nearest Neighbors** (KNeighborsClassifier with k=5)
3. **Logistic Regression** (with increased max_iter for convergence)
4. **Decision Tree** (DecisionTreeClassifier)
5. **Random Forest** (RandomForestClassifier)

### 3. Evaluation Metrics
For each model, the following metrics are calculated:
- Accuracy (both training and test sets)
- Precision
- Recall
- F1-Score
- ROC-AUC (where available)
- Cross-validation accuracy (5-fold on training set)

### 4. Visualization
- Comparative bar chart showing all metrics across models
- Individual confusion matrix heatmaps for each model

## Usage
1. Ensure the dataset path is correct in the code
2. Run the script to:
   - Train all models
   - Calculate performance metrics
   - Generate visual comparisons

## Expected Output
- A DataFrame comparing performance metrics across all models
- A bar chart visualizing model performance comparison
- Confusion matrix heatmaps for each individual model

## Key Functions
- `calculate_metrics()`: Comprehensive evaluation function that calculates multiple performance metrics and cross-validation scores
- Main execution workflow: Handles data loading, preprocessing, model training, and visualization

## Interpretation Guide
- **Accuracy**: Overall correctness of the model
- **Precision**: How many selected instances are relevant
- **Recall**: How many relevant instances are selected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes
- **Confusion Matrix**: Detailed breakdown of true/false positives/negatives

## Potential Improvements
1. Feature engineering and selection
2. Hyperparameter tuning for each model
3. Handling class imbalance if present
4. Additional model types (SVM, Gradient Boosting, etc.)
5. More detailed exploratory data analysis

## Notes
- The random state is fixed (42) for reproducibility
- Stratification ensures representative class distribution in train/test splits
- Min-Max scaling preserves the original distribution while normalizing the range

This project provides a solid foundation for binary classification problems and can be adapted to other medical diagnosis tasks or similar classification problems.

**Author**: Ali Zokaei  
**Libraries**: pandas, matplotlib, seaborn, scikit-learn