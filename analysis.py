import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.data")

# To output to a txt file analysis.py > analysis.txt was entered in the terminal. Need a better method
# Code needs to be entered after every 

# Having a look at the data set, checking that it loaded. 
print(f'The first five rows of the data set are: \n {iris.head()}')

# Add column names
iris.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
print(f'Checking that the column names are correct: \n {iris.head()}')

# Summary of the variables and data types in the data set
print(iris.info())

# Summary statistics of the data set
print(iris.describe())

# Summary statistics for each variable by flower species

setosa = iris['species'] == 'setosa'


iris['species'].unique()

setosa = iris[iris['species'] == 'Iris-setosa']
print(f'Summary statistics for Iris setosa are: \n{setosa.describe()}')
print()

versicolor = iris[iris['species'] == 'Iris-versicolor']
print(f'Summary statistics for Iris versicolor are: \n{versicolor.describe()}')
print()

virginica = iris[iris['species'] == 'Iris-virginica']
print(f'Summary statistics for Iris virginica are: \n{virginica.describe()}')
print()