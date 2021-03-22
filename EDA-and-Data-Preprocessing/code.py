# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(path)
print(data.shape)
#data['Rating'].hist()

#print(type(data['Rating'][1]))
#Code starts here
data = data[data['Rating']<=5]
print(data.shape)
data['Rating'].hist()

#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()
percent_null = total_null*100/data.isnull().count()
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])
print(missing_data)
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1*100/data.isnull().count()
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total', 'Percent'])
print(missing_data_1)
# code ends here


# --------------
#Code starts here
sns.catplot(x='Category', y='Rating', data=data, kind='box', height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#Code starts here
#print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace(',','')
#print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace('+','')
#print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].astype(int)
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
sns.regplot(x="Installs", y="Rating", data=data)
plt.title('Rating vs Installs [RegPlot]')
plt.show()

#Code ends here



# --------------
#Code starts here
#print(data['Price'].value_counts())

data['Price'] = data['Price'].str.replace('$', '')
data['Price'] = data['Price'].astype(float)
sns.regplot(x="Price", y="Rating", data=data)
plt.title('Rating vs Price [Regplot]')
plt.show()
#Code ends here


# --------------

#Code starts here

data['Genres'] = data['Genres'].str.split(';').str[0]

gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()
#print(gr_mean.head(5))
print(gr_mean.describe())
gr_mean = gr_mean.sort_values('Rating')
print(gr_mean.head(1))
print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here
#print(data['Last Updated'])
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
print(data['Last Updated'].head(3))
max_data = max(data['Last Updated'])
print(max_data)

data['Last Updated Days'] = max_data - data['Last Updated']
data['Last Updated Days'] = data['Last Updated Days'].dt.days

sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title("Rating vs Last Updated [RegPlot]")
plt.show()

#Code ends here


