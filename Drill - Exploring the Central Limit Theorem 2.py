# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

# Let’s make histograms for the two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1') 
plt.hist(pop2, alpha=0.5, label='Population 2') 
plt.legend(loc='upper right') 
plt.show()

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()

print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())

# Compute the difference between the two sample means.
diff=sample2.mean( ) -sample1.mean()
print(diff)

size = np.array([len(sample1), len(sample2)])
sd = np.array([sample1.std(), sample2.std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#The difference between the means divided by the standard error: T-value.  
print(diff/diff_se)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))

bigsample1 = np.random.choice(pop1, 1000, replace=True)
bigsample2 = np.random.choice(pop2, 1000, replace=True)

plt.hist(bigsample1, alpha=0.5, label='sample 1') 
plt.hist(bigsample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()

print(bigsample1.mean())
print(bigsample2.mean())
print(bigsample1.std())
print(bigsample2.std())

# Compute the difference between the two sample means.
diff=bigsample2.mean( ) -bigsample1.mean()
print(diff)

size = np.array([len(bigsample1), len(bigsample2)])
sd = np.array([bigsample1.std(), bigsample2.std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#The difference between the means divided by the standard error: T-value.  
print(diff/diff_se)

from scipy.stats import ttest_ind
print(ttest_ind(bigsample2, bigsample1, equal_var=False))

lilsample1 = np.random.choice(pop1, 20, replace=True)
lilsample2 = np.random.choice(pop2, 20, replace=True)

plt.hist(lilsample1, alpha=0.5, label='sample 1') 
plt.hist(lilsample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right') 
plt.show()

print(lilsample1.mean())
print(lilsample2.mean())
print(lilsample1.std())
print(lilsample2.std())

# Compute the difference between the two sample means.
diff=lilsample2.mean( ) -lilsample1.mean()
print(diff)

size = np.array([len(lilsample1), len(lilsample2)])
sd = np.array([lilsample1.std(), lilsample2.std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#The difference between the means divided by the standard error: T-value.  
print(diff/diff_se)

from scipy.stats import ttest_ind
print(ttest_ind(lilsample2, lilsample1, equal_var=False))

pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

# Let’s make histograms for the two groups. 

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))

pop1 = np.random.binomial(10, 0.4, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

# Let’s make histograms for the two groups. 

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))

pop1 = np.random.geometric(0.4, 10000)
pop2 = np.random.geometric(0.5, 10000) 



# Let’s make histograms for the two groups. 

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)
print(sample1.mean(),pop1.mean())
from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))