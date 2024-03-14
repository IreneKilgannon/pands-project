import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.data")

# Having a look at the data set, checking that it loaded. 
print(iris.head())

# Summary of the variables and data types in the data set
print(iris.info())


# Summary statistics of the data set
print(iris.describe())
