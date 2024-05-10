from exploratory import plot_hist


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data set, add headings to the columns.
iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

plot_hist(iris, hue = 'species')