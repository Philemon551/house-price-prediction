# 🏠 House Price Prediction Using Machine Learning

An end-to-end machine learning project that predicts residential house prices using the **Ames Housing** dataset.

The project covers:
- data inspection and cleaning
- missing-value analysis
- exploratory data analysis (EDA)
- outlier investigation
- feature engineering
- preprocessing with Scikit-Learn
- model training and comparison
- 5-fold cross-validation
- prediction on unseen houses

## Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook
- Git / GitHub

## Project Workflow

1. Load and inspect the Ames Housing data
2. Review missing values and distinguish true missing data from domain-specific values such as “No Garage” or “No Basement”
3. Perform exploratory analysis on the target and important numerical/categorical features
4. Investigate influential observations
5. Engineer additional housing features
6. Log-transform `SalePrice`
7. Split the data into training and test sets
8. Build a `ColumnTransformer` for numerical, ordinal and nominal predictors
9. Train Linear Regression, Decision Tree and Random Forest pipelines
10. Evaluate models with MAE, RMSE and R²
11. Perform 5-fold cross-validation
12. Compare predictions on randomly selected unseen houses

## Feature Engineering

The project creates several additional predictors:

- **TotalSF** — basement + first-floor + second-floor square footage
- **TotalBathrooms** — full and half bathrooms combined, including basement bathrooms
- **HouseAge** — year sold minus year built
- **RemodAge** — year sold minus year remodeled
- **HasGarage**
- **HasBasement**
- **HasFireplace**
- **HasPool**

## Target Transformation

`SalePrice` is right-skewed, so the model is trained on:

```python
LogSalePrice = np.log1p(SalePrice)
```

Predictions are converted back to the original price scale with:

```python
SalePrice = np.expm1(prediction)
```

## Preprocessing

The model uses a `ColumnTransformer`:

- numerical features → `StandardScaler`
- ordinal features → `OrdinalEncoder`
- nominal features → `OneHotEncoder`

Unknown categories are handled safely during inference.

## Holdout Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | $15,806 | $23,028 | 0.904 |
| Decision Tree | $27,828 | $41,621 | 0.686 |
| Random Forest | $16,011 | **$22,903** | **0.905** |

**Interpretation:** Linear Regression achieved the lowest holdout MAE, while Random Forest achieved the best RMSE and R².

## Cross-Validation Results

5-fold cross-validation was performed on the log-transformed target.

| Model | CV MAE | CV RMSE | CV R² | R² Std |
|---|---:|---:|---:|---:|
| Linear Regression | **0.0879** | **0.1269** | **0.8948** | 0.0247 |
| Decision Tree | 0.1515 | 0.2141 | 0.7034 | 0.0438 |
| Random Forest | 0.0939 | 0.1352 | 0.8830 | **0.0097** |

Linear Regression produced the strongest average cross-validation score, while Random Forest showed the lowest R² variability across folds.

## Unseen-House Check

A practical test was also performed on randomly selected houses from the unseen test set. Random Forest produced the lowest average prediction error in the project’s spot checks, including the 20-house comparison in the notebook.

## Recommended Visuals

After running the notebook, save or export these visuals to the `images/` folder:

1. `missing_values.png`
2. `saleprice_distribution.png`
3. `grlivarea_vs_saleprice.png`
4. `correlation_heatmap.png`
5. `saleprice_log_transform.png`
6. `actual_vs_predicted_linear.png`
7. `model_comparison.png`
8. `unseen_house_predictions.png`

The most important visuals for recruiters are the **model comparison**, **actual vs predicted**, and **correlation heatmap**.

## Repository Structure

```text
house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── PROJECT_REVIEW.md
├── notebooks/
│   └── house_price_prediction.ipynb
├── data/
│   └── README.md
└── images/
    └── README.md
```

## How to Run

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Ames Housing `train.csv` file and place it inside the `data/` folder.
5. Open the notebook and run the cells from top to bottom.

> The cleaned notebook expects the dataset at `../data/train.csv`.

## Key Findings

- `OverallQual` and `GrLivArea` show strong positive relationships with `SalePrice`.
- Neighborhood also has a strong relationship with selling price.
- Feature engineering improves the representation of total usable space, bathrooms and property age.
- A single Decision Tree generalizes worse than Linear Regression and Random Forest.
- Random Forest achieved the best holdout RMSE and R².
- Linear Regression remained highly competitive and produced the best average cross-validation performance.

## Possible Improvements

- Move all statistically learned imputations fully inside the training pipeline to eliminate any remaining leakage risk.
- Tune Random Forest hyperparameters with `GridSearchCV` or `RandomizedSearchCV`.
- Compare Ridge, Lasso, Elastic Net, Gradient Boosting and XGBoost/LightGBM if permitted.
- Add feature-importance or permutation-importance analysis.
- Save the selected trained pipeline with `joblib`.
- Deploy a small prediction interface with Streamlit or Flask.

## Author

**Thankgod Philemon Akor**  
Electrical & Electronics Engineering Graduate | Data Science & Machine Learning

GitHub: [philemon551](https://github.com/philemon551)
