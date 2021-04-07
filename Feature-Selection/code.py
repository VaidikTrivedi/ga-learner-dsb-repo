# --------------
import pandas as pd

dataset = pd.read_csv(path)
print(dataset.head(5))
dataset.drop('Id', axis=1, inplace=True)
print(dataset.describe())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns
#print(cols)

#number of attributes (exclude target)
size = len(cols.drop('Cover_Type'))
print(size)

#x-axis has target attribute to distinguish between classes
x = dataset['Cover_Type']

#y-axis shows values of an attribute
y = dataset.drop('Cover_Type', axis=1)

#Plot violin for all attributes
for i in range (0, size):
    ax = sns.violinplot(x=dataset[cols[i]])
    plt.show()


# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
#Continous variable subset
subset_train = dataset.iloc[:,:10]

#Correlation beetween variables
data_corr = subset_train.corr(method='pearson')

#Heatmap of correlatiton
sns.heatmap(data_corr)
plt.show()

#Sorting according to value
correlation = data_corr.unstack().sort_values(kind='quicksort')

#Slicing
corr_var_list = correlation[((correlation>upper_threshold) | (correlation<lower_threshold)) & (correlation != 1)]
print(corr_var_list)

# Code ends here


# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
X = dataset.drop('Cover_Type', axis=1)
Y = dataset['Cover_Type']

#Split Data
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
scaler = StandardScaler()

#Standardized
#scaler.fit(X_train, Y_train)

#Apply transform only for continuous data
X_train_temp = scaler.fit_transform(X_train.iloc[:, :10])
X_test_temp = scaler.transform(X_test.iloc[:, :10])

#Concatenate scaled continuous data and categorical
X_train1 = np.concatenate((X_train_temp, X_train.iloc[:, 10:dataset.shape[1]-1]), axis=1)
X_test1 = np.concatenate((X_test_temp, X_test.iloc[:, 10:dataset.shape[1]-1]), axis=1)

scaled_features_train_df = pd.DataFrame(data=X_train1, index=X_train.index, columns=X_train.columns)
scaled_features_test_df = pd.DataFrame(data=X_test1, index=X_test.index, columns=X_test.columns)


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

# Write your solution here:
skb = SelectPercentile(score_func=f_classif, percentile=90)

#Fit and Transform
predictors = skb.fit_transform(X_train1, Y_train)
scores = skb.scores_

Features = X_train.columns.values
#print(Features)
#Create dataframe
dataframe = pd.DataFrame(data=[Features, scores])
dataframe = dataframe.T
dataframe.columns = ['Features', 'scores']
#print(dataframe.head(5))

#Sort DataFrame
dataframe.sort_values('scores', ascending=False, inplace=True)
top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])
print(top_k_predictors)





# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = LogisticRegression()
clf1 = OneVsRestClassifier(clf)
model_fit_all_features = clf1.fit(X_train, Y_train)
predications_all_features = clf1.predict(X_test)
score_all_features = accuracy_score(Y_test, predications_all_features)
print("score_all_features: ", score_all_features)
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features = clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test, predictions_top_features)
print("score_top_features: ", score_top_features)


