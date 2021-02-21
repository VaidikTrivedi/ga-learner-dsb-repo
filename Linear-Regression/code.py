# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(path)
print(df.head(5))

X = df.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]].values
#print(X)
y = df['list_price']
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)
#print(X_train, X_test, y_train, y_test)


# --------------
import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns
#print(y)
#print(cols)
fig, axes = plt.subplots(nrows=3, ncols=3)
for i in range(0, 3):
    for j in range(0, 3):
        col = cols[i*3 + j]
        #print(X_train[col])
        plt.scatter(X_train[col], y_train)
        plt.xlabel(col)
        plt.show()
# code ends here



# --------------
# Code starts here
corr = X_train.corr()
#print(corr)
X_train.drop(columns = ['play_star_rating', 'val_star_rating'], inplace=True)
#X_train.drop('val_star_rating', axis=1)
X_test.drop('play_star_rating', axis=1, inplace=True)
X_test.drop('val_star_rating', axis=1, inplace=True)
print(X_train, X_test)
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse, "\nr2: ", r2)

# Code ends here


# --------------
# Code starts here
residual = y_test - y_pred
#print(residual)
plt.hist(residual, bins = 10)
plt.show()

# Code ends here


