import pandas as pd

# read data 
# 1 means test, 0 means control
data = pd.read_Cav("./test_basic.csv")

# sanity check
print(data.head())


# t-test
from scipy import stats

print(data.groupby('test')['conversion'].mean())

test = stats.ttest_ind(data.loc[data['test'] == 1]['conversion'], data.loc[data['test'] == 0]['conversion'], equal_var=False)

print(test.statistic)

print(test.pvalue)


#print test results
if (test.pvalue>0.05):
  print ("Non-significant results")
elif (test.statistic>0):
  print ("Statistically better results")
else:
  print ("Statistically worse results")

# sample size
import statsmodels.stats.api as sms

p1_and_p2 = sms.proportion_effectsize(0.1, 0.11)

print("The required sample size per group is ", round(sample_size))  

import numpy as np
import matplotlib.pyplot as plt
#Possible p2 values. We choose from 10.5% to 15% with 0.5% increments
possible_p2 = np.arange(.105, .155, .005)
print(possible_p2)


sample_size = []
for i in possible_p2:
   p1_and_p2 = sms.proportion_effectsize(0.1, i)
   sample_size.append(sms.NormalIndPower().solve_power(p1_and_p2, power=0.8, alpha=0.05))
plt.plot(sample_size, possible_p2)
plt.title("Sample size vs Minimum Effect size")
plt.xlabel("Sample Size")
plt.ylabel("Minimum Test Conversion rate")
plt.show()