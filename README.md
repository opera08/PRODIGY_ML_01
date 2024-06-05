# House Price Prediction Using XGBoost

This repository contains code for predicting house prices using the XGBoost regression model. The dataset used is from a housing price competition, and the model aims to predict the sale price of houses based on various features.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   ```
2. Change to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data

The dataset consists of training data (`train.csv`), test data (`test.csv`), and a sample submission file (`sample_submission.csv`). Ensure these files are located in the project directory.

## Data Preprocessing

1. Load the data:
   ```python
   import pandas as pd

   houseprice_data = pd.read_csv('train.csv')
   test_data = pd.read_csv('test.csv')
   sample_data = pd.read_csv('sample_submission.csv')
   ```

2. Select relevant columns:
   ```python
   columns_to_add = houseprice_data.iloc[:, [0, 1, 3, 4]]
   result_df = pd.concat([sample_data, columns_to_add], axis=1)
   houseprice_data['SalePrice'] = sample_data['SalePrice']
   ```

3. Impute missing values and drop rows with missing target values:
   ```python
   result_df['LotFrontage'] = result_df['LotFrontage'].fillna(result_df['LotFrontage'].median())
   result_df = result_df.dropna(subset=['SalePrice'])
   ```

4. Split the data into features (`X`) and target (`y`):
   ```python
   X = result_df.drop(['SalePrice'], axis=1)
   y = result_df['SalePrice']
   ```

5. Encode categorical variables:
   ```python
   X_encoded = pd.get_dummies(X)
   ```

## Exploratory Data Analysis

1. Calculate the correlation matrix and visualize it using a heatmap:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   correlation = result_df.corr()
   plt.figure(figsize=(10, 10))
   sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 7}, cmap='Purples')
   plt.show()
   ```

## Model Training

1. Split the data into training and testing sets:
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
   ```

2. Ensure unique column names in the training data:
   ```python
   def make_unique_columns(df):
       columns = pd.Series(df.columns)
       for dup in columns[columns.duplicated()].unique():
           dup_indices = columns[columns == dup].index.tolist()
           for i, idx in enumerate(dup_indices):
               if i > 0:
                   new_col_name = f"{dup}_{i}"
                   while new_col_name in columns:
                       i += 1
                       new_col_name = f"{dup}_{i}"
                   columns[idx] = new_col_name
       df.columns = columns
       return df

   X_train = make_unique_columns(X_train)
   ```

3. Train the XGBoost model:
   ```python
   from xgboost import XGBRegressor

   model = XGBRegressor()
   model.fit(X_train, Y_train)
   ```

## Evaluation

1. Predict the training data and evaluate the model:
   ```python
   from sklearn import metrics

   training_data_prediction = model.predict(X_train)
   score_1 = metrics.r2_score(Y_train, training_data_prediction)
   score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
   print("R Squared Error:", score_1)
   print("Mean Absolute Error:", score_2)
   ```

## Visualization

1. Plot the actual prices vs. predicted prices:
   ```python
   plt.scatter(Y_train, training_data_prediction)
   plt.xlabel("Actual Prices")
   plt.ylabel("Predicted Prices")
   plt.title("Actual Price vs Predicted Price")
   plt.show()
   ```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
