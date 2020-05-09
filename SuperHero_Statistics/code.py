# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
#Code starts here 
#print(data.head(5))
data['Gender'].replace('-', 'Agender', inplace = True)
gender_count = data['Gender'].value_counts()
print("Gender Counts: ", gender_count)
plt.bar(['Male', 'Female', 'Agender'], gender_count)
plt.show()


# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
plt.pie(alignment, labels = data['Alignment'].value_counts().keys())
plt.title('Character Alignment')
plt.show()


# --------------
#Code starts here
sc_df = data[['Strength', 'Combat']].copy()
sc_covariance = sc_df.cov()['Strength']['Combat']
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)
print("Strength vs Combat:\n", sc_pearson)


ic_df = data[['Intelligence', 'Combat']]
ic_covariance = ic_df.cov()['Intelligence']['Combat']
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_pearson = ic_covariance/ (ic_intelligence*ic_combat)
print("Intelligence vs Combat:\n", ic_pearson)


# --------------
#Code starts here
total_high = data['Total'].quantile(.99)
super_best = data[data['Total']>total_high]
super_best_names = [super_best['Name']]
print(super_best_names)


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize = (20, 8))

ax_1.boxplot(super_best['Intelligence'])
ax_1.set(title = 'Intelligence')
ax_2.boxplot(super_best['Speed'])
ax_2.set(title = 'Speed')
ax_3.boxplot(super_best['Power'])
ax_3.set(title = 'Power')
#plt.show()


