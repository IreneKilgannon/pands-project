import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("iris.data", header = None)

# To output to a txt file analysis.py > analysis.txt was entered in the terminal. Need a better method
# Code needs to be entered after every 

# Having a look at the data set, checking that it loaded. 
print(f'The first five rows of the data set are: \n {iris.head()}')
print()

# Add column names
iris.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
print(f'Checking that the column names are correct: \n {iris.head()}')
print()

# Summary of the variables and data types in the data set
print(f'Summary of the variables and the data types in the data set.')
print(iris.info())
print()

# Looking for missing data, NaN
print(f'Checking to see if there is any missing data or NaN. \n{iris.isna().sum()}')
print()

# Summary statistics of the data set
print(f'Overall summary statistics for the data set. \n{iris.describe()}')
print()


# Summary statistics for each variable by flower species

print(f"The unique names in the species column are: \n {iris['species'].unique()}")
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

# Save a histogram of each variable to png files
for col in iris:
    sns.set_palette("Set1")
    sns.histplot(x = col, data = iris, hue = 'species', bins = 20)
    plt.title(f'Histogram of {col}')
    plt.savefig(f'{col}.png')
    plt.show()

# Outputs a scatter plot of each pair of variables
sns.pairplot(iris, hue = 'species')
#fig.suptitle('Scatter plot of each variable in the dataset')
plt.savefig('Scatter_plot.png')
#plt.show()