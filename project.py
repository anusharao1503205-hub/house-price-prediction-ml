import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


data = {
    'Size': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 4, 5],
    'Age': [10, 5, 12, 8, 2],
    'Price': [300000, 400000, 500000, 550000, 700000]
}
df = pd.DataFrame(data)


X = df.drop('Price', axis=1)
y = df['Price']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, predictions)}")
print(f"R2 Score: {r2_score(y_test, predictions)}")

new_house = [[2000, 3, 10]] 
print(f"Predicted Price: ${model.predict(new_house)[0]:,.2f}")