# --------------
import pandas as pd
import numpy as np

data = pd.read_csv(path)
sample_size=2000
data_sample = data.sample(n=sample_size, random_state=0)
#print(data_sample.head(5))
sample_mean = data_sample['installment'].mean()
print("Sample Mean: ",sample_mean)
sample_std = data_sample['installment'].std()
print("Sample Std: ",sample_std)
z_critical = 23.7105
margin_of_error = z_critical * np.sqrt(sample_std/sample_size)
print("Margin of Error: ", margin_of_error)
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(confidence_interval)
true_mean = data['installment'].mean()
print("True Mean: ",true_mean)
if(true_mean>confidence_interval[0] and true_mean<confidence_interval[1]):
    print("Falls in Range")
else:
    print("Out of Range")


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig, axes = plt.subplots(nrows = 3, ncols = 1)
for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        m.append(data['installment'].sample(n = sample_size[i]).mean())
    #print(m)
    mean_series = pd.Series(m)
    axes[i].plot(mean_series)



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].str.rstrip('%').astype('float')/100
#print(data.head(5))
z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'], value=data['int.rate'].mean(), alternative='larger')
print(z_statistic, p_value)
if(p_value<0.05):
    print('Hypothesis Accepted')
else:
    print('Hypothesis Not Acceptable')


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value = ztest(data[data['paid.back.loan']=='No']['installment'],
                 data[data['paid.back.loan']=='Yes']['installment'])

print(round(z_statistic, 2), round(p_value, 2))

if(p_value>0.05):
    print('Null hypothesis is not acceptable')
else:
    print('Null hypothesis is acceptable')


# --------------
#Importing header files
from scipy.stats import chi2_contingency
from  scipy import stats

#Critical value
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data['purpose'][data['paid.back.loan']=='Yes'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()
print('Value Count of Yes: ', yes.head(5))
print('Value Count of No: ', no.head(5)) 
observed = pd.concat([yes.transpose(), no.transpose()], 1, keys=['Yes', 'No'])
print(observed['Yes'][0])
print(observed['No'][0])
chi2, p, dof, ex = chi2_contingency(observed)
print(chi2, p, dof)
if(chi2>critical_value):
    print('Hypothesis not Acceptable')
else:
    print('Hypothesis Acceptable')


