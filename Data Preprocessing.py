# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2) Read the dataset located here 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
# 3) Assign new headers to the DataFrame
data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']
print('\n=====================================================')
print(data)

# 4) Drop the 'Sample code number' attribute 
data = data.drop(['Sample code number'],axis=1)
print('\n=====================================================')
print(data)

### Missing Values ###
# 5)Convert the '?' to NaN
data = data.replace('?',np.NaN)

# 6) Count the number of missing values in each attribute of the data.
print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))
    
# 7) Discard the data points that contain missing values
data = data.dropna()
print('\n=====================================================')
print(data)

### Outliers ### 
# 8)Draw a boxplot to identify the columns in the table that contain outliers 
# Columns that contain outliers:
#    Marginal Adhesion, Single Epithelial Cell Size,
#           Bland Chromatin, Normal Nucleoli, Mitoses
data.boxplot(figsize=(30,25))
plt.show()


### Duplicate Data ### 
# 9) Check for duplicate instances.
dups = data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))

# 10) Drop row duplicates
print('Number of rows before discarding duplicates = %d' % (data.shape[0]))
data = data.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (data.shape[0]))
print('\n=====================================================')
print(data)

### Discretization ### 
# 11) Plot a 10-bin histogram of the attribute values 'Clump Thickness' distribution
data['Clump Thickness'].hist(bins=10)

# 12) Discretize the Clump Thickness' attribute into 4 bins of equal width.
data['Clump Thickness'] = pd.cut(data['Clump Thickness'], 4)
print(data['Clump Thickness'].value_counts(sort=False))


### Sampling ### 
# 13) Randomly select 1% of the data without replacement. The random_state argument of the function specifies the seed value of the random number generator.
sample = data.sample(frac=0.01, random_state=1, replace=False)
print('\n', sample)