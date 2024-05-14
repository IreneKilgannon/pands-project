import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotting as pt

# Set up seaborn colour palette. #5A4FCF is the colour code for iris. 
colors = ['#5A4FCF', '#4ECF99', '#CF4E99']
sns.set_palette(colors)

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

# Value count of each species.
count_species = f"A count of each species: \n {iris['species'].value_counts()} \n\n"

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
    f.write(count_species)
    f.write(summary_statistics)
    f.write(setosa_summary)
    f.write(versicolor_summary)
    f.write(virginica_summary)


# A box plot to visually compare the summary statistics across the three species in the data set.

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
plt.savefig('plots\\Box_plot.png')
plt.close()

######################################
# HISTOGRAM CODE

# TASK: Save a histogram of each variable to png files #####

# Create a histogram of each numeric variable as a figure axes plot. 
fig, ax = plt.subplots(2, 2, figsize = (13, 13))

# Histogram of sepal length
sns.histplot(iris, x = 'sepal_length', ax = ax[0,0])
ax[0, 0].set_title('Histogram of Sepal Length')
ax[0, 0].set_xlabel('Sepal Length (cm)')

# Histogram of sepal width
sns.histplot(iris, x = 'sepal_width', ax = ax[0, 1])
ax[0, 1].set_title('Histogram of Sepal Width')
ax[0, 1].set_xlabel('Sepal Width (cm)')

# Histogram of petal length
sns.histplot(iris, x = 'petal_length', ax = ax[1, 0])
ax[1, 0].set_title('Histogram of Petal Length')
ax[1, 0].set_xlabel('Petal Length (cm)')

# Histogram of petal width
sns.histplot(iris, x = 'petal_width', ax = ax[1, 1])
ax[1, 1].set_title('Histogram of Petal Width')
ax[1, 1].set_xlabel('Petal Width (cm)')

# Overall title for the plot
plt.suptitle('Histogram of the Iris Data Set')

# Histogram saved as png file
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Summary_Histogram.png')
plt.close()


# Creating a function to iterate through the numeric columns in the data set.


# Call the plot_hist function from the plotting module on the iris data set.
pt.plot_hist(iris, hue = 'species')

######################################
# SCATTER PLOT CODE

#### Output a scatter plot of each pair of variables ####

# Call the plot_scatter function from the plotting module on the iris data set.
pt.plot_scatter(iris, hue = 'species')


# Use a pair plot! Much simplier method to generate a scatter plot of each pair of variables
g = sns.pairplot(iris, hue = 'species')
g.fig.suptitle('Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('plots\\Pair_plot.png')
plt.close()



#### Any other analysis ####

# To calculate the correlation coefficient between two variables
corr_SL_vs_SW = iris['sepal_length'].corr(iris['sepal_width'])
print(f'The correlation coefficient between sepal length and sepal width is {corr_SL_vs_SW.round(3)}')

# Create a correlation matrix between the numeric variables in the data set.
correlation_matrix = iris.drop(['species'], axis = 1).corr()
print(correlation_matrix)


# Create a heatmap of the correlation coefficients between the variables in the data set.
fig, ax = plt.subplots(2, 2, figsize = (15, 12))

# Overall values  - not taking the flower species into account
sns.heatmap(iris.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[0, 0], vmin = -0.5, vmax=1)
ax[0,0].set_title('Overall')

# Iris setosa
sns.heatmap(setosa.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[0, 1], vmin = -0.5, vmax=1)
ax[0,1].set_title('Iris setosa')

# Iris versicolor
sns.heatmap(versicolor.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[1, 0], vmin = -0.5, vmax=1)
ax[1,0].set_title('Iris versicolor')

# Iris virginica
sns.heatmap(virginica.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[1,1], vmin = -0.5, vmax=1)
ax[1,1].set_title('Iris virginica')

# Add title
plt.suptitle('Correlation Coefficients for the Iris Data Set')
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Heatmap_correlation_coefficients.png')
plt.close()

##### Regression Plots for Selected Variables

sepal_length_array = iris['sepal_length'].to_numpy()

sepal_width_array = iris['sepal_width'].to_numpy()

# Use numpy polyfit to fit a straight line between x and y.
# np.polyfit(x-axis, y-axis, deg). Deg = 1 for a linear equation.
m, c = np.polyfit(sepal_length_array, sepal_width_array, 1)

# Return values for the slope, m and y-intercept, c.
print(f'The value of the slope is {m.round(3)}.')
print(f'The value of the intercept is {c.round(3)}.')



# Demonstrating how to plot a regression line on a scatter plot using numpy.
fig, ax = plt.subplots()

# A scatter plot of Sepal Width vs sepal length using the numpy array generated in the previous cell.
ax.scatter(sepal_length_array, sepal_width_array)

# Plotting the trend line. The y-axis values are generated from the equation of the line, with m and c equal to the values generated above.
ax.plot(sepal_length_array, m * sepal_length_array + c, '#CF4E99')

# Axis labels.
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Title.
plt.title('Sepal Width vs Sepal Length')
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Numpy_reg_plot.png')
plt.close()
# 
fig, ax = plt.subplots(2, 2, figsize = (15, 10))

# Regression plot between sepal length and sepal width
sns.regplot(iris, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 0], ci = None)

# Regression plot between sepal width and sepal length by species
sns.regplot(setosa, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 1], label = 'setosa', ci = None)
sns.regplot(versicolor, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 1], label = 'versicolor', ci = None)
sns.regplot(virginica, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 1], label = 'virginica', ci = None)

# Regression plot between petal length and petal width
sns.regplot(iris, x = 'petal_length', y = 'petal_width', ax = ax[1, 0], ci = None)

# Regression plot between petal length and petal width by species
sns.regplot(setosa, x = 'petal_length', y = 'petal_width', ax = ax[1, 1], label = 'setosa', ci = None)
sns.regplot(versicolor, x = 'petal_length', y = 'petal_width', ax = ax[1, 1], label = 'versicolor', ci = None)
sns.regplot(virginica, x = 'petal_length', y = 'petal_width', ax = ax[1, 1], label = 'virginica', ci = None)

# Add title
plt.suptitle('Some Regression Plots for Iris Data Set')
ax[0, 0].set_title('Sepal Width vs Sepal Length')
ax[0, 1].set_title('Sepal Width vs Sepal Length by Species')
ax[1, 0].set_title('Petal Width vs Petal Length')
ax[1, 1].set_title('Petal Width vs Petal Length by Species')

# Set x-axis labels
ax[0, 0].set_xlabel('Sepal Length (cm)')
ax[0, 1].set_xlabel('Sepal Length (cm)')
ax[1, 0].set_xlabel('Petal Length (cm)')
ax[1, 1].set_xlabel('Petal Length (cm)')

# Set y-axis labels
ax[0, 0].set_ylabel('Sepal Width (cm)')
ax[0, 1].set_ylabel('Sepal Width (cm)')
ax[1, 0].set_ylabel('Petal Width (cm)')
ax[1, 1].set_ylabel('Petal Width (cm)')


# Save plots
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Regression_plots.png')
plt.close()

# Pair regression plot
sns.pairplot(iris, hue = 'species', kind = 'reg')
plt.suptitle('Regression Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Pair_Regression_plots.png')
plt.close()

# lmplot example. Sepal Width vs Sepal Length
sns.lmplot(iris, x = 'sepal_length', y = 'sepal_width', ci = None, col = 'species')
plt.suptitle('Sepal Width vs Sepal Length by Species', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\lmplot_example.png')
plt.close()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

# Instantiate the model
reg = LinearRegression()

# Select the columns of interest from the dataset
X = iris['petal_length'].values
y = iris['petal_width'].values

# 
X = X.reshape(-1, 1)

# Split the data into training set and test set data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 47)

# Fit the training data to the model
reg.fit(X_train, y_train)

# Predict the y data points by using the predict function on the X_test data.
y_pred = reg.predict(X_test)

# Print out the predictions and the actual values of the y_test data.
print(f'Predictions: {y_pred[:5]}, Actual values: {y_test[:5]}')

# r_squared measures the accuracy of the results.
r_squared = reg.score(X_test, y_test)

#
rmse = mean_squared_error(y_test, y_pred, squared= False)

print(f"R^2: {r_squared}")
print(f"RMSE: {rmse}")
print(f'slope:' , reg.coef_)
print(f'intercept:' , reg.intercept_)

# Scatter plot of petal width vs petal lenght and line plot of the predicted values.
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color = '#CF4E99')

# Label the x-axis and y-axis
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\lg_analysis.png')
plt.close()



kf = KFold(n_splits = 5, shuffle = True, random_state=47)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv = kf)

print(f'cv results are {cv_results}')
print(np.mean(cv_results))
print(np.std(cv_results))
print(np.quantile(cv_results, [0.025, 0.975]))

