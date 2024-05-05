import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up seaborn palette of 'Set1'
sns.set_palette("Set1")

# Import the data set, add headings to the columns.
iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Check that the data set has loaded. 
head = f'The first five rows of the data set are: \n {iris.head()}\n \n'

# Open and write the head of the data set to analysis.txt.
with open('analysis.txt', 'wt') as f:
    f.write(head)

#### Output a summary of each variable to a single txt file, analysis.txt ####

# Collating the necessary infomation for analysis.txt

# Get the number of rows and columns in the data set.
shape = f'The shape of the data set is {iris.shape}. \n\n'

# Get the variable names.
column_names = f'Summary of the variable names in the data set are: \n {iris.columns} \n\n'

# Get the data types of the variables.
data_types = f'The data types in the data set are: \n{iris.dtypes}\n \n'

# Look for missing data, NaN
missing_values = f'Checking to see if there is any missing data or NaN. \n{iris.isna().sum()} \n \n'

# Uniques names in the species column.
unique = f"The unique names in the species column are: \n {iris['species'].unique()} \n\n"

# Summary statistics for the overall the data set
summary_statistics = f'Overall summary statistics for the data set. \n{iris.describe()} \n\n'

# Create dataframes for each iris species.
setosa = iris[iris['species'] == 'Iris-setosa']
versicolor = iris[iris['species'] == 'Iris-versicolor']
virginica = iris[iris['species'] == 'Iris-virginica']


# Summary Statistics for Iris setosa
setosa_summary = f'Summary statistics for Iris setosa are: \n{setosa.describe()} \n\n'

# Summary Statistics for Iris versicolor
versicolor_summary = f'Summary statistics for Iris versicolor are: \n{versicolor.describe()} \n\n'

# Summary Statistics for Iris virginica
virginica_summary = f'Summary statistics for Iris virginica are: \n{virginica.describe()} \n\n'

# Write the summary statistics to analysis.txt
# Append them to the analysis.txt file created previously

with open('analysis.txt', 'a') as f:
    f.write(shape)
    f.write(column_names)
    f.write(data_types)
    f.write(missing_values)
    f.write(unique)
    f.write(summary_statistics)
    f.write(setosa_summary)
    f.write(versicolor_summary)
    f.write(virginica_summary)


# Create a box plot to visually compare the summary statistics across the three species in the data set.

# Create a fig, ax plot
fig, ax = plt.subplots(2,2, figsize = (10, 10))

# Create a box plot for each variable, coloured by species.
sns.boxplot(ax = ax[0, 0], x = 'species', y = 'sepal_length', data = iris)
sns.boxplot(ax = ax[0, 1], x = 'species', y= 'sepal_width', data = iris)
sns.boxplot(ax = ax[1, 0], x = 'species', y = 'petal_length', data = iris)
sns.boxplot(ax = ax[1, 1], x = 'species', y = 'petal_width', data = iris)

# Overall plot title
plt.suptitle('Box plot by Species for Each Variable')

# Label each plot
ax[0,0].set_title('Sepal Length')
ax[0,1].set_title('Sepal Width')
ax[1,0].set_title('Petal Length')
ax[1,1].set_title('Petal Width')

# Save the created plot
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Box_plot.png')
plt.close()

#### Save a histogram of each variable to png files #####

# Save a histogram of each variable to png files.
# Creating a function that iterateds through the data set, would ideally like to generalise more so that it could in theory be used for any data set.

def plot_hist(df):
    for x in df:
        #sns.set_palette("Set1")
        sns.histplot(x = x, data = df, hue = 'species')
        plt.title(f"Histogram of {x.title().replace('_', ' ')}")
        plt.xlabel(f"{x.replace('_', ' ')}")
        plt.savefig(f'C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Histogram_of_{x}.png')
        #plt.show()
        plt.close()

plot_hist(iris)

#### Output a scatter plot of each pair of variables ####

# Creating a function

# How would I improve it, 
    # 1. generalise it more, if x or y are an object data type, skip 
    # 2. only 1 of plot of each eg currently getting petal length vs petal width and then petal width vs petal length. One plot would do.
    # 

def plot_scatter(df):
    for x in df:
        for y in df:
            # Do not create a scatter plot for the following conditions
            if x == y or x == 'species' or y == 'species':
                continue
            else:
                # Create a scatter plot
                sns.scatterplot(data = df, x = x, y = y, hue = 'species')

                # Add title to plot
                plt.title(f"Scatter plot of {y.title().replace('_', ' ')} vs {x.title().replace('_', ' ')}")

                # Label x and y-axis
                plt.xlabel(f"{x.title().replace('_', ' ')}")
                plt.ylabel(f"{y.title().replace('_', ' ')}")

                # Save a scatter plot for each pair of variables
                plt.savefig(f'C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Scatterplot_{y}_vs_{x}.png')
                plt.close()

plot_scatter(iris)

# Use a pair plot! Much simplier method to generate a scatter plot of each pair of variables
g = sns.pairplot(iris, hue = 'species')
g.fig.suptitle('Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Pair_plot.png')
plt.close
#plt.show()


#### Any other analysis ####

# Create a correlation matrix between the numeric variables in the data set.
correlation_matrix = iris.drop(['species'], axis = 1).corr()



# Create a heatmap of the correlation coefficients between the variables in the data set.
fig, ax = plt.subplots(2, 2, figsize = (12, 10))

# Overall values  - not taking the flower species into account
sns.heatmap(iris.drop(['species'], axis = 1).corr(), annot = True, ax = ax[0, 0], vmin = -0.5, vmax=1)
ax[0,0].set_title('Overall')

# Iris setosa
sns.heatmap(setosa.drop(['species'], axis = 1).corr(), annot = True, ax = ax[0, 1], vmin = -0.5, vmax=1)
ax[0,1].set_title('Iris setosa')

# Iris versicolor
sns.heatmap(versicolor.drop(['species'], axis = 1).corr(), annot = True, ax = ax[1, 0], vmin = -0.5, vmax=1)
ax[1,0].set_title('Iris versicolor')

# Iris virginica
sns.heatmap(virginica.drop(['species'], axis = 1).corr(), annot = True, ax = ax[1,1], vmin = -0.5, vmax=1)
ax[1,1].set_title('Iris virginica')

# Add title
plt.suptitle('Correlation Coefficients for the Iris Data Set')
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Heatmap_correlation_coefficients.png')
plt.close()
