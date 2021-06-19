# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
#print(df.head(5))
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges'] = X_train['TotalCharges'].replace(' ', np.NaN)
X_test['TotalCharges'] = X_test['TotalCharges'].replace(' ', np.NaN)

X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

X_train.fillna('NaN', inplace=True)
X_test.fillna('NaN', inplace=True)

#print(X_train.isnull().sum())
continous = ['customerID','MonthlyCharges', 'TotalCharges', 'tenure']
categorical =  [x for x in X_train.columns if x not in continous]
#print(categorical)

X_train[categorical] = X_train[categorical].apply(LabelEncoder().fit_transform)
X_test[categorical] = X_test[categorical].apply(LabelEncoder().fit_transform)

y_train.replace({'No':0, 'Yes':1}, inplace=True)
y_test.replace({'No':0, 'Yes':1}, inplace=True)


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)

ada_score = accuracy_score(y_test, y_pred)
print(f"Accuracy Rate: {ada_score*100} %")

ada_cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: ", ada_cm)

ada_cr = classification_report(y_test, y_pred)
print("Classification Report: ", ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

xgb_score = accuracy_score(y_test, y_pred)
print(f"Accuracy Rate: {xgb_score*100} %")

xgb_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", xgb_cm)

xgb_cr = classification_report(y_test, y_pred)
print("Classification Report: ", xgb_cr)

clf_model = GridSearchCV(estimator=xgb_model, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

clf_score = accuracy_score(y_test, y_pred)
print(f"Accuracy Rate: {clf_score*100} %")

clf_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", clf_cm)

clf_cr = classification_report(y_test, y_pred)
print("Classification Report: ", clf_cr)


