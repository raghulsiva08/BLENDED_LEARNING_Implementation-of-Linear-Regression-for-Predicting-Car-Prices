# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## REF NO:25015705
### DATE:08-02-2026
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and prepare data – Import dataset, select predictor variables (enginesize, horsepower, citympg, highwaympg) and target (price).
2.Split dataset – Divide into training and testing sets using train_test_split.
3.Scale features – Standardize predictors with StandardScaler for better regression performance.
4.Train model & predict – Fit LinearRegression on training data, generate predictions on test data.
5.Evaluate & visualize – Compute metrics (MSE, RMSE, R², MAE), check residuals (Durbin‑Watson, homoscedasticity), and plot actual vs predicted prices.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv('CarPrice_Assignment.csv')

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred=model.predict(x_test_scaled)

print('Name: RAGHUL.S')
print('Reg no: 212225040325')
print("MODEL COEFFICIENTS: ")
for feature,coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'intercept':>12}: {model.intercept_:10}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test,y_pred)):>10}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10}")
print(f"{'ABSOLUTE':>12}: {mean_absolute_error(y_test,y_pred):>10}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(),y.max()],'r--')
plt.title("Linearity check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals= y_test - y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Raghul.s
RegisterNumber: 212225040325
*/
```

## Output:
![WhatsApp Image 2026-02-08 at 9 30 53 PM](https://github.com/user-attachments/assets/50472dfc-c9b7-4757-b523-f5368d62dbf8)

![WhatsApp Image 2026-02-08 at 9 30](https://github.com/user-attachments/assets/5ba40c00-6900-4706-82aa-a96a5615c3fc)

![price](https://github.com/user-attachments/assets/e87fed78-cf71-4f02-be82-4b91017f4e4c)

![WhatsApp Image 2026-02-08 at 9 30 51 PM](https://github.com/user-attachments/assets/cebea0a7-0c4a-42ee-afd3-e2a2ee0cb847)

![WhatsApp Image 2026-02-08 at 9 30 49 PM](https://github.com/user-attachments/assets/e483f411-0b47-4c34-ac7e-cc40efa30146)

<img width="1459" height="570" alt="Screenshot 2026-02-08 220619" src="https://github.com/user-attachments/assets/a763fe4b-d247-4ae9-bb1a-90c13d3acef9" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
