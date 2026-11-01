# House Price Prediction using Linear Regression

A Python project implementing Multiple Linear Regression to estimate residential real estate prices based on square footage, bedroom count, and bathroom count.

## Features
- Data cleaning and preprocessing using `pandas`.
- Model training and split evaluation using `scikit-learn`.
- Key performance metrics evaluation: MAE, RMSE, and $R^2$ score.
- Model interpretation (intercept and regression coefficients).
- Command line interface (CLI) to easily run on any compatible dataset.

## Dataset
This script is configured for the **Kaggle House Prices - Advanced Regression Techniques** dataset (`train.csv`), utilizing:
- `GrLivArea`: Above grade living area in square feet.
- `BedroomAbvGr`: Number of bedrooms above ground level.
- `FullBath`: Number of full bathrooms above ground level.
- `SalePrice`: Property sale price (Target variable).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Usage

Run the model training script:
```bash
python house_price_prediction.py --data train.csv
```

## License
MIT
