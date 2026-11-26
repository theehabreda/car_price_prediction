# Car Price Prediction (MSRP)

Predict a car's list price (MSRP) from its features (make, year, engine power, fuel economy, body style, etc.) using machine learning. The project covers the full workflow: data cleaning, EDA, leakage-free feature engineering, model comparison with cross-validation, error analysis, and a saved model ready for deployment.



## Problem

Given a car's specifications, estimate its MSRP in US dollars. Prices have a long right tail, so the model predicts `log(MSRP + 1)` and the result is converted back to dollars.

## Dataset

*Car Features and MSRP* from Kaggle (about 12k rows, features such as make, model, year, engine HP and cylinders, fuel type, transmission, drive type, body style, MPG, popularity).


- Download the CSV file and save it as `data_car_price.csv` in the project folder.

## Approach

1. **Cleaning:** normalize column names and text values, and drop duplicate rows (duplicates can leak between train and test and inflate scores).
2. **EDA:** price distribution (raw vs log), missing values, correlations, price vs engine power and year, median price by make.
3. **Split:** train / validation / test = 60 / 20 / 20. The test set is used **once**, at the end.
4. **Features (`CarFeatures` transformer):**
   - numeric missing values filled with the **training-set median**;
   - `age = reference_year - year` (reference year = newest year in the training data);
   - one-hot encoding for the number of doors and for the 5 most frequent values of each categorical column.

   Everything learned from data is learned in `fit()` on training rows only, so there is no data leakage, and the same object works inside a scikit-learn `Pipeline` and inside cross-validation.
5. **Models compared:**
   - Ridge regression **implemented from scratch** with NumPy (closed-form solution), checked against scikit-learn's `Ridge`;
   - Ridge (scikit-learn), Random Forest, Gradient Boosting (`HistGradientBoostingRegressor`).
6. **Selection:** 5-fold cross-validation on train + validation; the model with the lowest mean CV RMSE is re-trained on train + validation and evaluated once on the test set.
7. **Analysis:** predicted vs actual plot, worst predictions, and permutation importance of the raw features.

## Results

Metrics are reported on the validation set, then cross-validated, then on the test set. RMSE and R² are on the log scale; MAE is in dollars; median APE is the typical percentage error.

**Validation set**

| Model | RMSE (log) | MAE ($) | Median APE (%) | R² (log) |
|---|---|---|---|---|
| Ridge (from scratch) | 0.508 | 13159.342 | 25.499 | 0.767 |
| Ridge (scikit-learn) | 0.508 | 13159.342 | 25.499 | 0.767 |
| Random Forest | 0.126 | 4142.293 | 6.674 | 0.986 |
| Gradient Boosting | 0.116 | 4187.688 | 6.478 | 0.988 |

**5-fold cross-validation (RMSE, log scale)**

| Model | Mean | Std |
|---|---|---|
| Ridge (scikit-learn) | 0.4743 | 0.0189 |
| Random Forest | 0.1196 | 0.0101 |
| Gradient Boosting | 0.1124 | 0.0067 |

**Final model on the test set:** Gradient Boosting, with 

RMSE (log) : 0.109, 

MAE : 4361.053 $, 

median APE : 6.108 %, 

R² : 0.989.



## Project structure

```
.
├── car_price_prediction_clean.ipynb   # full workflow, run top to bottom
├── data_car_price.csv                 # dataset (download separately)
├── car_price_model.joblib             # saved model (created by the notebook)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook car_price_prediction_clean.ipynb
```

Run all cells from top to bottom. The last cells save the trained pipeline to `car_price_model.joblib`.

## Predicting a new car

Inside the notebook, `predict_price` takes a dictionary with the dataset's column names:

```python
car = {
    "make": "ford", "model": "mustang", "year": 2015,
    "engine_fuel_type": "regular_unleaded", "engine_hp": 300,
    "engine_cylinders": 6, "transmission_type": "manual",
    "driven_wheels": "rear_wheel_drive", "number_of_doors": 2,
    "market_category": "performance", "vehicle_size": "midsize",
    "vehicle_style": "coupe", "highway_mpg": 28, "city_mpg": 19,
    "popularity": 5657,
}
print(predict_price(car))   # estimated MSRP in dollars
```

Text values use lowercase with underscores (the same cleaning the notebook applies to the dataset).

## Limitations

- The data ends in 2017, so `age` is relative to that year and the model is not meant for current-year prices.
- MSRP is the manufacturer's list price, not the resale or market price.
- Only the 5 most frequent values of each categorical column are kept; rarer values are ignored.



## Author

**Ehab Reda**: Junior AI Engineer, Cairo, Egypt
LinkedIn: www.linkedin.com/in/theehabreda · GitHub: github.com/theehabreda

