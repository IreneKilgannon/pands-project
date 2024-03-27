import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.data", header = None)

# To output to a txt file analysis.py > analysis.txt was entered in the terminal. Need a better method
# Code needs to be entered after every 

# Having a look at the data set, checking that it loaded. 
print(f'The first five rows of the data set are: \n {iris.head()}')

# Add column names
iris.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
print(f'Checking that the column names are correct: \n {iris.head()}')

# Summary of the variables and data types in the data set
print(iris.info())
print()

# Summary statistics of the data set
print(iris.describe())
print()

# Summary statistics for each variable by flower species


print(iris['species'].unique())
print()

setosa = iris[iris['species'] == 'Iris-setosa']
print(f'Summary statistics for Iris setosa are: \n{setosa.describe()}')
print()

versicolor = iris[iris['species'] == 'Iris-versicolor']
print(f'Summary statistics for Iris versicolor are: \n{versicolor.describe()}')
print()

virginica = iris[iris['species'] == 'Iris-virginica']
print(f'Summary statistics for Iris virginica are: \n{virginica.describe()}')
print()

# Histogram of Sepal Length by species. 
fig, ax = plt.subplots(1, 3, sharex= True, sharey= True)
ax[0].hist(setosa['sepal_length_cm'], label = 'Iris setosa', edgecolor = 'black', alpha = 0.3, bins = 15)
ax[1].hist(versicolor['sepal_length_cm'], label = 'Iris versicolor', edgecolor = 'black', alpha = 0.3, bins = 15)
ax[2].hist(virginica['sepal_length_cm'], label = 'Iris virginica', edgecolor = 'black', alpha = 0.3, bins = 15)
#ax.suptitle('Histogram of sepal length for Iris species')
#ax.set_xlabel('Sepal length (cm)')
#ax.set_ylabel('No of occurrances')
#ax.legend()
plt.savefig('sepal_length.png')
plt.show()

def iris_hist(**kwargs):
    '''A function that accepts user inputs to plot a histogram for the iris data set'''
    iris_dict = {}
