import pandas
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.width', 350)
  
#read from google drive
data= pandas.read_csv("./novelty_effect.csv")
print(data.head())

from scipy import stats
#t-test of test vs control for our target metric 
test = stats.ttest_ind(data.loc[data['test'] == 1]['pages_visited'], data.loc[data['test'] == 0]['pages_visited'], equal_var=False)
  
#t statistics
print(test.statistic)

print(test.pvalue)

ab_test_old = stats.ttest_ind(data.loc[(data['test'] == 1) & (data['signup_date']!=data['test_date'])]['pages_visited'], 
                              data.loc[(data['test'] == 0) & (data['signup_date']!=data['test_date'])]['pages_visited'], 
                              equal_var=False)
#t statistics
print(ab_test_old.statistic)

print(ab_test_old.pvalue)

#new users
ab_test_new = stats.ttest_ind(data.loc[(data['test'] == 1) & (data['signup_date']==data['test_date'])]['pages_visited'], 
                              data.loc[(data['test'] == 0) & (data['signup_date']==data['test_date'])]['pages_visited'], 
                              equal_var=False)
#t statistics
print(ab_test_new.statistic)
print(ab_test_new.pvalue)


