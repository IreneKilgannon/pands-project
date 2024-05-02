import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("iris.data", header = None)

# Having a look at the data set, checking that it loaded. 
head = f'The first five rows of the data set are: \n {iris.head()}\n \n'

with open('analysis.txt', 'wt') as f:
    f.write(head)

# Add column names
iris.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
add_header = f'Checking that the column names are correct: \n {iris.head()}\n \n'

with open('analysis.txt', 'a') as f:
    f.write(add_header)

# Summary of the variables and data types in the data set
info = f'Summary of the variables and the data types in the data set {iris.info()} \n\n'

data_types = f'The data types in the data set are: \n{iris.dtypes}\n \n'

# Looking for missing data, NaN
missing_values = f'Checking to see if there is any missing data or NaN. \n{iris.isna().sum()} \n \n'


# Uniques names of flower species
unique = f"The unique names in the species column are: \n {iris['species'].unique()} \n\n"


# Summary statistics of the data set
summary_statistics = f'Overall summary statistics for the data set. \n{iris.describe()} \n\n'


# Creating dataframes for each iris species.
setosa = iris[iris['species'] == 'Iris-setosa']
versicolor = iris[iris['species'] == 'Iris-versicolor']
virginica = iris[iris['species'] == 'Iris-virginica']


# Summary Statistics for Iris setosa
setosa_summary = f'Summary statistics for Iris setosa are: \n{setosa.describe()} \n\n'


# Summary Statistics for Iris versicolor
versicolor_summary = f'Summary statistics for Iris versicolor are: \n{versicolor.describe()} \n\n'


# Summary Statistics for Iris virginica
virginica_summary = f'Summary statistics for Iris virginica are: \n{virginica.describe()} \n\n'

# Writing summary statistics to analysis.txt
with open('analysis.txt', 'a') as f:
    f.write(info)
    f.write(data_types)
    f.write(missing_values)
    f.write(unique)
    f.write(summary_statistics)
    f.write(setosa_summary)
    f.write(versicolor_summary)
    f.write(virginica_summary)


# Comparing summary statistics across the three species.
fig, axes = plt.subplots(2,2, figsize = (8, 8))
sns.boxplot(ax = axes[0, 0], x = 'species', y = 'sepal_length_cm', data = iris)
sns.boxplot(ax = axes[0, 1], x = iris['species'], y= iris['sepal_width_cm'])
sns.boxplot(ax = axes[1, 0], x = iris['species'], y = iris['petal_length_cm'])
sns.boxplot(ax = axes[1, 1], x = iris['species'], y = iris['petal_width_cm'])
plt.suptitle('Summary Statistics by Species for Each Variable')
axes[0,0].set_title('Sepal Length')
axes[0,1].set_title('Sepal Width')
axes[1,0].set_title('Petal Length')
axes[1,1].set_title('Petal Width')
plt.savefig('Boxplot.png')
#plt.show()


# Save a histogram of each variable to png files
for col in iris:
    sns.set_palette("Set1")
    sns.histplot(x = col, data = iris, hue = 'species')
    plt.title(f'Histogram of {col}')
    plt.savefig(f'{col}.png')
    plt.show()
    plt.close()

# Outputs a scatter plot of each pair of variables
sns.pairplot(iris, hue = 'species')
#fig.suptitle('Scatter plot of each variable in the dataset')
plt.savefig('Scatter_plot.png')
#plt.show()


# Heatmap of the Correlation Coefficients
fig, ax = plt.subplots(2, 2, figsize = (12, 10))

sns.heatmap(iris.drop(['Species'], axis = 1).corr(), annot = True, ax = ax[0, 0], vmin = -0.5, vmax=1)
sns.heatmap(setosa.drop(['Species'], axis = 1).corr(), annot = True, ax = ax[0, 1], vmin = -0.5, vmax=1)
sns.heatmap(versicolor.drop(['Species'], axis = 1).corr(), annot = True, ax = ax[1, 0], vmin = -0.5, vmax=1)
sns.heatmap(virginica.drop(['Species'], axis = 1).corr(), annot = True, ax = ax[1,1], vmin = -0.5, vmax=1)
plt.savefig('Heatmap_correlation_coefficients.png')
plt.show()