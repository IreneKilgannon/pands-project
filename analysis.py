import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotting as pt

# For linear regression analysis the following modules are required:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

# To ignore a warning regarding a change in the figure layout of the seaborn plots.
# UserWarning: The figure layout has changed to tight self._figure.tight_layout(*args, **kwargs)
import warnings
warnings.filterwarnings('ignore')

# Set up seaborn colour palette. #5A4FCF is the colour code for the colour iris. 
colors = ['#5A4FCF', '#4ECF99', '#CF4E99']
sns.set_palette(colors)

# Import the data set, add headings to the columns.
iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Check that the data set has loaded by displaying the first five rows. 
head = f'The first five rows of the data set are: \n {iris.head()}\n \n'

# Create analysis.txt and write the head of the data set to it.
with open('analysis.txt', 'wt') as f:
    f.write(head)

#### Output a summary of each variable to a single txt file, analysis.txt ####
# Collating the necessary infomation for analysis.txt

#### Overall information about the data set

# Get the number of rows and columns in the data set.
shape = f'The shape of the data set is {iris.shape}. \n\n'

# Get the variable names.
column_names = f'Summary of the variable names in the data set are: \n {iris.columns} \n\n'

# Get the data types of the variables.
data_types = f'The data types in the data set are: \n{iris.dtypes}\n \n'

# Look for missing data, NaN
missing_values = f'Checking to see if there is any missing data or NaN. \n{iris.isna().sum()} \n \n'

#### Summary information for the species column

# Uniques names in the species column.
unique = f"The unique names in the species column are: \n {iris['species'].unique()} \n\n"

# Value count of each species.
count_species = f"A count of each species: \n {iris['species'].value_counts()} \n\n"

#### Summary information for the numeric columns

# Summary statistics for the overall data set
summary_statistics = f'Overall summary statistics for the data set. \n{iris.describe()} \n\n'

# Summary statistics grouped by species. Transpose the result to for an easier read. 
summary_by_species = f"Summary statistics grouped by species \n{iris.groupby('species').describe().transpose()} \n\n"

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
    f.write(summary_by_species)
    f.write(setosa_summary)
    f.write(versicolor_summary)
    f.write(virginica_summary)

######################################
# A box plot to visually compare the summary statistics across the three species in the data set.

# Create a fig, ax plot
fig, ax = plt.subplots(2,2, figsize = (10, 10))

# Create a box plot for each variable, coloured by species.
sns.boxplot(ax = ax[0, 0], x = 'species', y = 'sepal_length', data = iris)
sns.boxplot(ax = ax[0, 1], x = 'species', y = 'sepal_width', data = iris)
sns.boxplot(ax = ax[1, 0], x = 'species', y = 'petal_length', data = iris)
sns.boxplot(ax = ax[1, 1], x = 'species', y = 'petal_width', data = iris)

# Overall plot title
plt.suptitle('Box plot of Each Variable by Species')

# Label each subplot title
ax[0,0].set_title('Sepal Length')
ax[0,1].set_title('Sepal Width')
ax[1,0].set_title('Petal Length')
ax[1,1].set_title('Petal Width')

# Label y-axis
ax[0,0].set_ylabel('Sepal Length (cm)')
ax[0,1].set_ylabel('Sepal Width (cm)')
ax[1,0].set_ylabel('Petal Length (cm)')
ax[1,1].set_ylabel('Petal Width (cm)')

# Save the box plot
plt.savefig('plots\\Box_plot.png')
plt.close()

######################################
# HISTOGRAM CODE

# TASK: Save a histogram of each variable to png files

# Create a histogram of each numeric variable as a figure axes plot with two rows and two columns.
fig, ax = plt.subplots(2, 2, figsize = (13, 13))

# Histogram of sepal length, with title and x-axis label
sns.histplot(iris, x = 'sepal_length', ax = ax[0,0])
ax[0, 0].set_title('Histogram of Sepal Length')
ax[0, 0].set_xlabel('Sepal Length (cm)')

# Histogram of sepal width, with title and x-axis label
sns.histplot(iris, x = 'sepal_width', ax = ax[0, 1])
ax[0, 1].set_title('Histogram of Sepal Width')
ax[0, 1].set_xlabel('Sepal Width (cm)')

# Histogram of petal length, with title and x-axis label
sns.histplot(iris, x = 'petal_length', ax = ax[1, 0])
ax[1, 0].set_title('Histogram of Petal Length')
ax[1, 0].set_xlabel('Petal Length (cm)')

# Histogram of petal width, with title and x-axis label
sns.histplot(iris, x = 'petal_width', ax = ax[1, 1])
ax[1, 1].set_title('Histogram of Petal Width')
ax[1, 1].set_xlabel('Petal Width (cm)')

# Overall title for the plot
plt.suptitle('Histogram of the Iris Data Set')

# Histogram saved as png file
plt.savefig('plots\\Summary_Histogram.png')
plt.close()

#####     #####     #####
# Histograms by species with the plotting.py module.

# Call the plot_hist function from the plotting module on the iris data set.
# Hue is the species column to separate the species by colour. 
pt.plot_hist(iris, hue = 'species')

#####     #####     #####
# Histograms for petal length and petal width for Iris setosa, the number of bins are important.
fig, ax = plt.subplots(1, 2)

# Create the histograms
sns.histplot(setosa, x = 'petal_length', ax = ax[0])
sns.histplot(setosa, x = 'petal_width', ax = ax[1])

# Label the axis
ax[0].set_xlabel('Petal Length (cm)')
ax[1].set_xlabel('Petal Width (cm)')

# Add title, save the plot
plt.suptitle('Histograms of Petal Length and Width for Iris setosa')
plt.savefig('plots\\Hist_Setosa_pl.png')
plt.close()

######################################
# SCATTER PLOT CODE

# Use of a pairplot.
sns.pairplot(iris)

# Save the plot
plt.savefig('plots\\Pair_plot.png')
plt.close()

#####     #####     #####

# Call the plot_scatter function from the plotting module.
pt.plot_scatter(iris, hue = 'species')

#####     #####     #####
# Calculating the limits for outliers for the sepal width of Iris setosa

# Minimium value in Sepal Width columns for Iris setosa
min = setosa['sepal_width'].min()

# Identifying outliers
# Calculating the range for outliers for the sepal width for Iris setosa.

# Calculate the 75th percentile
seventy_fifth = setosa['sepal_width'].quantile(0.75)

# Calculate the 25th percentile.
twenty_fifth = setosa['sepal_width'].quantile(0.25)

# IQR (interquartile range) Difference between the 75th and 25th percentile.
s_width_iqr = seventy_fifth - twenty_fifth

# Upper Outliers, points outside the 75th percentile plus 1.5 times the IQR.
upper_limit = seventy_fifth + (1.5 * s_width_iqr)

# Lower Outliers, points outside the 25th percentile minus 1.5 times the IQR.
lower_limit = twenty_fifth - (1.5 * s_width_iqr)

with open('analysis.txt', 'a') as f:
    f.write(f'The minimium value in the sepal width column for Iris setosa is {min}\n')
    f.write(f'The lower limit for outliers in the sepal width column for Iris setosa is {lower_limit.round(2)}.\n')
    f.write(f'The upper limit for outliers in the sepal width column for Iris setosa is {upper_limit.round(2)}.\n')


######################################
#### ANY OTHER ANALYSIS

# Creating a regression line on a scatter plot using numpy

# Create a numpy array of the sepal length and sepal width columns
sepal_length_array = iris['sepal_length'].to_numpy()
sepal_width_array = iris['sepal_width'].to_numpy()

# Use numpy polyfit to fit a straight line between x and y.
# np.polyfit(x-axis, y-axis, deg). Deg = 1 for a linear equation.
m, c = np.polyfit(sepal_length_array, sepal_width_array, 1)

# Write the values for the slope, m and y-intercept, c to analysis.txt.
with open('analysis.txt', 'a') as f:
    f.write('Calculating the m and c values of the line with numpy.\n')
    f.write(f'The value of the slope for sepal width vs sepal length is {m.round(3)}.\n')
    f.write(f'The value of the intercept for sepal width vs sepal length is {c.round(3)}.\n\n')

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
plt.savefig('plots\\Numpy_reg_plot.png')
plt.close()

#####     #####     ##
# Pair regression plot 
sns.pairplot(iris, kind = 'reg')
plt.suptitle('Regression Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('plots\\Pair_Regression_plots.png')
plt.close()

#####     #####     #####
# To calculate the correlation coefficient between two variables
corr_SL_vs_SW = iris['sepal_length'].corr(iris['sepal_width'])

#####     #####     #####
# Create a correlation matrix between the numeric variables in the data set.
correlation_matrix = iris.drop(['species'], axis = 1).corr()

with open('analysis.txt', 'a') as f:
    f.write(f'The correlation coefficient between sepal length and sepal width is {corr_SL_vs_SW.round(3)}.\n\n')
    f.write(f'The correlation matrix for the variables in the iris data set. \n{correlation_matrix}\n\n')

#####     #####     #####
# Create a heatmap of the correlation coefficients between the variables in the data set.
fig, ax = plt.subplots(2, 2, figsize = (15, 12))

# Overall values - not taking the flower species into account
sns.heatmap(iris.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[0, 0], vmin = -0.5, vmax=1, cmap = 'Purples')
ax[0,0].set_title('Overall')

# Iris setosa
sns.heatmap(setosa.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[0, 1], vmin = -0.5, vmax=1, cmap = 'Purples')
ax[0,1].set_title('Iris setosa')

# Iris versicolor
sns.heatmap(versicolor.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[1, 0], vmin = -0.5, vmax=1, cmap = 'Purples')
ax[1,0].set_title('Iris versicolor')

# Iris virginica
sns.heatmap(virginica.drop(['species'], axis = 1).corr(), annot = True, linewidths = 0.2, ax = ax[1,1], vmin = -0.5, vmax=1, cmap = 'Purples')
ax[1,1].set_title('Iris virginica')

# Add title
plt.suptitle('Correlation Coefficients for the Iris Data Set')
plt.savefig('plots\\Heatmap_correlation_coefficients.png')
plt.close()

#####     #####     #####
# Regression plots for selected variables.

# Set up a figure, axes plot of 2 rows and 2 columns
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
plt.suptitle('Regression Plots for Selected Variables in the Iris Data Set')
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

# Add legend
plt.legend()

# Save plot
plt.savefig('plots\\Regression_plots.png')
plt.close()

#####     #####     #####
# Linear regression analysis

# Instantiate the model
reg = LinearRegression()

# Select the columns of interest from the dataset, X is the feature, y is the target variable
X = iris['petal_length'].values
y = iris['petal_width'].values

# Reshape the X data from a 1-D array to a 2-D array.
X = X.reshape(-1, 1)

# Split the data into training set and test set data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 47)

# Fit the training data to the model
reg.fit(X_train, y_train)

# Predict the y data points by using the predict function on the X_test data.
y_pred = reg.predict(X_test)

# Print out the predictions and the actual values of the y_test data.
with open('analysis.txt', 'a') as f:
    f.write('The first five predicted values for petal width and the actual values are:\n')
    f.write(f'Predicted Values: {y_pred[:5].round(3)}\nActual Values: {y_test[:5]}\n\n')

# R_squared for the test data
r_squared_test = reg.score(X_test, y_test)

# R_squared for the training data
r_squared_train = reg.score(X_train, y_train)

# Calculate root mean square error.
rmse = mean_squared_error(y_test, y_pred, squared = False)

# Coefficient for the regresssion line
coefficent = reg.coef_

# Intercept of the regression line
intercept = reg.intercept_

# To manually calculate RMSE
n = len(y_pred)
# Finish the manual calculation of the MSE
manual_rmse = np.sqrt(sum((y_test - y_pred)**2) / n)

# Write the variables to analysis.txt
with open('analysis.txt', 'a') as f:
    f.write('Performance of the linear regression model.\n')
    f.write(f"The value of R-squared for the test data: {r_squared_test.round(3)}.\n")
    f.write(f"The value of R-squared for the training data: {r_squared_train.round(3)}.\n")
    f.write(f"The RMSE is : {rmse.round(3)}.\n")
    f.write(f'RMSE calculated manually is {manual_rmse.round(3)}.\n')
    f.write(f'The slope of the regression line for petal width vs petal length is: {coefficent.round(3)}.\n')
    f.write(f'The intercept of the regression line for petal width vs petal length is {intercept.round(3)}.\n\n')

# Scatter plot of petal width vs petal length
plt.scatter(X_train, y_train)

# Line plot of the predicted values.
plt.plot(X_test, y_pred, color = '#CF4E99')

# Add title, label the x-axis and y-axis
plt.title('Linear regression analysis plot for petal width vs petal length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.savefig('plots\\lg_analysis.png')
plt.close()

#####     #####     #####
# Plotting residuals

# Calculate the residual, observed value minus predicted value. 
residuals = y_test - y_pred

# Plotting residuals
sns.residplot(x = y_pred, y = residuals)

# Add title
plt.title('Residuals plot')
# Label x-axis
plt.xlabel('Fitted values, y_test')
#Label y-axis
plt.ylabel('Residuals')
# Save the plot
plt.savefig('plots\\residuals.png')
plt.close()

#####     #####     #####
# K-Fold cross validation to calculate R-squared for differents splits of the data set.

# Set up the k_fold parameters
kf = KFold(n_splits = 5, shuffle = True, random_state=47)

# Specify regression model
reg = LinearRegression()

# Run cross_val_score, the score default is R-squared
cv_results = cross_val_score(reg, X, y, cv = kf)

# Write the results of k-fold analysis to analysis.txt
with open('analysis.txt', 'a') as f:
    f.write('k-Fold analysis results.\n')
    f.write(f'The value of R-squared for petal width vs petal length for each fold are {cv_results.round(3)}\n')
    f.write(f'The mean of R-squared is {np.mean(cv_results).round(3)}\n')
    f.write(f'The standard deviation for R-squared is {np.std(cv_results).round(3)}\n')
    f.write(f'The 95% quantile limits are {np.quantile(cv_results, [0.025, 0.975]).round(3)}\n\n')