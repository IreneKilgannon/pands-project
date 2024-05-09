
![Banner](https://github.com/IreneKilgannon/pands-project/blob/main/Fisher%E2%80%99s%20Iris%20Data%20Set.png)

Author: Irene Kilgannon

This is my analysis of the Fisher's Iris data set for the programming and scripting module.

## Making a plan
1. Using a readme for discussion  ~~decided~~.
2. background research add some images
3. write summary statistics to .txt file -  ~~done~~. Discussion next
4. histograms - created a function, Create a module with all the plotting files?. Add discussion
5. scatter plot /pair plot between all the variables. Add discussion
6. perhaps some linear regression plots. Make a start today.
7. calculate correlation coefficients, ~~fix labels on heatmap~~. add discussion
8. regression analysis https://campus.datacamp.com/courses/introduction-to-regression-with-statsmodels-in-python/simple-linear-regression-modeling?ex=1
9. machine learning sklearn


Ideally I would create modules, with all the functions required for the file and import them. Having trouble with this. Probably classes would be better but I don't understand, know enough about them!

## Project Statement
* Research the data set and summarise it in a README.
* Download the data set and add it to my repository.
* Write a program called analysis.py: 
    1. outputs a summary of each variable to a single text file
    2. saves a histogram of each variable to png files
    3. outputs a scatter plot of each pair of variables
    4. any other analysis - suggested machine learning or regression analysis?? LEARN about them.

## How to install

## How to use it

## Files in this project
analysis.py

analysis.ipynb

iris.data

README.md

plots directory with all the plots generated for this analysis

# Background to Fisher's Iris Data Set

In 1928 Edgar Anderson published his paper entitled ['The Problem of Species in the Northern Blue Flags, _Iris versicolor_ and _Iris virginica_'](https://www.biodiversitylibrary.org/page/15997721). Anderson was a evolutionary biologist interested in answering two questions namely, what are species and how have they originated? Between 1923 and 1928 he and his team studied _Iris versicolor_, at a number of different sites from Ontario in Canada to Alabama in the United States, by measuring a number of different iris characteristics. Surprisingly his study found that there were actually two different iris species present, _Iris versicolor_ and _Iris virginia_ and that it was possible to differentiate between them by geographic location. This is reflected in the [common names of these two species of iris](https://hgic.clemson.edu/factsheet/rain-garden-plants-iris-versicolor-and-iris-virginica/). _Iris versicolor_ is commonly known as the Northern blue flag iris and _Iris virginica_ is commonly known as the Southern blue flag iris.

ADD IMAGE TO BREAK UP THE TEXT

The data set is commonly known as Fisher's Iris Data set after the statistician and biologist, Ronald Fisher. The data measurements for _Iris setosa_ and _Iris versicolor_ were collected by Anderson from the same colony of plants in the Gaspé Peninsula, Quebec in 1935. According to [Unwin and Kleinman](hhttps://academic.oup.com/jrssig/article/18/6/26/7038520) the _Iris virginica_ data samples were from Anderson's original research and were collected in Camden, Tennessee. Fisher collated and analysed the data and in 1936 published his results in the Annals of Eugenics [The Use of Multiple Measurements in Taxonomic Problems](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x). He used a statistical method, linear discriminant analysis to attempt to distinguish the different iris species from each other. He found that _Iris setosa_ was easily distinguishable from the other two iris species using this method. 

![Image from Fisher's paper](https://github.com/IreneKilgannon/pands-project/blob/main/fisher_data_set_image.png)

Fisher's data set can viewed in his published paper but, in our computer age, the data set is available to download at [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). The data set is very widely used with currently over 700,000 views of the data set on the UCI website.

Fisher and racism  -do i put it in?

https://profjoecain.net/what-is-wrong-ronald-aylmer-fisher/

https://www.newstatesman.com/long-reads/2020/07/ra-fisher-and-science-hatred

https://www.nature.com/articles/s41437-020-00394-6

## Import the Required Modules.

Four modules are required for this analysis:

* [pandas](https://www.w3schools.com/python/pandas/pandas_intro.asp) - for manipulating data and for performing data analysis.
* [numpy](https://www.w3schools.com/python/numpy/default.asp) - performs a wide variety of mathematical calculations on arrays.
* [matplotlib.pyplot](https://www.geeksforgeeks.org/python-introduction-matplotlib/) - used to create plots e.g. bar plots, scatter plots, histograms.
* [seaborn](https://realpython.com/python-seaborn/) - a python data visualisation library based on matplotlib. Usually requires less code syntax than matplotlib.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load the Data Set

The data set was downloaded from [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) and imported. This csv file does not contain column names and the column names were obtained from the variables table on the [information page of the iris data set](https://archive.ics.uci.edu/dataset/53/iris). They column names are sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm and species.

A number of methods were explored to add the column names [adding headers to a dataframe in pandas a comprehensive-guide](https://saturncloud.io/blog/adding-headers-to-a-dataframe-in-pandas-a-comprehensive-guide/). The simplest method is to add the column names using the name parameter when the data set was loaded.

```python
iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
```

## Summary of the Data Set
***

[Analysis.txt](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.txt) contains the output of [analysis.py](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.py) and has provided the information used to summarise the data set.

f.write will only write strings to a txt file. [This reddit post](https://www.reddit.com/r/learnpython/comments/12emhsa/how_do_i_save_the_output_of_the_python_code_as_a/) suggested saving the output to a variable and then result could be written to a txt file.  IMPROVE 


<details>
<summary>Code written to analysis.txt</summary>

```python
 Collating the necessary infomation for analysis.txt

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
```
</details>
<br>

It is a small data set with 150 rows and five columns with each row corresponding to a different flower sample. There are three different iris species, _Iris setosa_, _Iris versicolor_ and _Iris virginica_ with 50 samples for each species. There is no missing data.

![iris](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YYiQed4kj_EZ2qfg_imDWA.png)

The five attributes(variables) in the data set are:
* sepal_length (measured in cm)
* sepal_width (measured in cm)
* petal_length (measured in cm)
* petal_width (measured in cm)
* species

A basic description of the structure of an iris flower will help to understand the variable names. Each iris has three true petals and three sepals. The three petals are upright and are also known as standards. Sepals are a modified leaf and are usually green in colour and its function is to protect the developing flower bud. When the flower has bloomed the iris' sepal is described as "the landing pad for bumblebees" by the [US Forest Service](https://www.fs.usda.gov/wildflowers/beauty/iris/flower.shtml). This diagram nicely illustrates the difference between the petals and the sepals and also how the width and length of each were measured.

![Length vs Width](https://www.integratedots.com/wp-content/uploads/2019/06/iris_petal-sepal-e1560211020463.png)

 
## Summary Statistics for the data set

Summary statistics for numeric data are easily obtained using the [describe method](https://www.w3schools.com/python/pandas/ref_df_describe.asp). It gives a count of each variable and some statistics such as the min, max, mean and std deviation for each variable.

```python
# Summary statistics for the overall the data set
summary_statistics = f'Overall summary statistics for the data set. \n{iris.describe()} \n\n'
```

Overall summary statistics for the data set.
|      | sepal_length | sepal_width | petal_length | petal_width|
|---|---|---|---|---|
|count |   150.000000 |  150.000000  |  150.000000  | 150.000000|
|mean  |     5.843333 |    3.054000  |    3.758667  |   1.198667|
|std   |     0.828066 |    0.433594  |    1.764420  |   0.763161|
|min   |     4.300000 |    2.000000  |    1.000000  |   0.100000|
|25%   |     5.100000 |    2.800000  |    1.600000  |   0.300000|
|50%   |     5.800000 |    3.000000  |    4.350000  |   1.300000|
|75%   |     6.400000 |    3.300000  |    5.100000  |   1.800000|
|max   |     7.900000 |    4.400000  |    6.900000  |   2.500000| 


<details>
<summary>Summary Statistics for Each Iris Species</summary>

Summary statistics for Iris setosa are:

|      | sepal_length | sepal_width | petal_length | petal_width|
|---|---|---|---|---|
|count  |    50.00000  |  50.000000   |  50.000000   |  50.00000|
|mean   |     5.00600  |   3.418000   |   1.464000   |   0.24400|
|std    |     0.35249  |   0.381024   |   0.173511   |   0.10721|
|min    |     4.30000  |   2.300000   |   1.000000   |   0.10000|
|25%    |     4.80000  |   3.125000   |   1.400000   |   0.20000|
|50%    |     5.00000  |   3.400000   |   1.500000   |   0.20000|
|75%    |     5.20000  |   3.675000   |   1.575000   |   0.30000|
|max    |     5.80000  |   4.400000   |   1.900000   |   0.60000| 


Summary statistics for Iris versicolor are: 
|      | sepal_length | sepal_width | petal_length | petal_width|
|---|---|---|---|---|
|count  |   50.000000  |  50.000000   |  50.000000  |  50.000000|
|mean   |    5.936000  |   2.770000   |   4.260000  |   1.326000|
|std    |    0.516171  |   0.313798   |   0.469911  |   0.197753|
|min    |    4.900000  |   2.000000   |   3.000000  |   1.000000|
|25%    |    5.600000  |   2.525000   |   4.000000  |   1.200000|
|50%    |    5.900000  |   2.800000   |   4.350000  |   1.300000|
|75%    |    6.300000  |   3.000000   |   4.600000  |   1.500000|
|max    |    7.000000  |   3.400000   |   5.100000  |   1.800000| 


Summary statistics for Iris virginica are: 
|      | sepal_length | sepal_width | petal_length | petal_width|
|---|---|---|---|---|
|count   |   50.00000   | 50.000000  |   50.000000   |  50.00000|
|mean    |    6.58800   |  2.974000  |    5.552000   |   2.02600|
|std     |    0.63588   |  0.322497  |    0.551895   |   0.27465|
|min     |    4.90000   |  2.200000  |    4.500000   |   1.40000|
|25%     |    6.22500   |  2.800000  |    5.100000   |   1.80000|
|50%     |    6.50000   |  3.000000  |    5.550000   |   2.00000|
|75%     |    6.90000   |  3.175000  |    5.875000   |   2.30000|
|max     |    7.90000   |  3.800000  |    6.900000   |   2.50000|

</details>

<br>

A [seaborn boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html) is nice visual method to display summary statistics about the data set. It gives us most of the same information that the describe method does but as it is visual it is easier and quicker to make comparisons. One difference between the describe method and a box plot is that describe gives us the mean, whereas a box plot displays the median value of the variable of interest.

![diagram to explain box plot](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/1_boxplots.jpg)

_Different parts of a boxplot | Image: Michael Galarnyk_


<details>
<summary>Box plot code</summary>

```python
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
```
</details>


![Boxplot](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Box_plot.png)



## Histogram of each variable saved to png files

Histograms are used to plot continouous numeric data. The four variables of sepal length, sepal width, petal length and petal width fulfill this criteria.

Rather than create a plot for each of the variables by coding each of the variables seperately, I wrote function to plot each of the variables in the data set by looping through the column names, which I called x. The histograms were created using a [seaborn histplot](). Seaborn's hue parameter made it very easy to differentiate each of the variables by the categorical variable, species. 

Currently the function is creating a histogram of the species column __NEED TO REFINE THE CODE SO THAT A HISTOGRAM OF SPECIES IS NOT CREATED__ Would also like to refine the code so that it could be used for any data set. The hue parameter is set to species for this data set but it would be nice if the user could select their own variable for hue. 

<details>
<summary>Histogram code</summary>

```python
def plot_hist(df):
    for x in df:
        sns.histplot(x = x, data = df, hue = 'species')
        plt.title(f"Histogram of {x.title().replace('_', ' ')}")
        plt.xlabel(f"{x.replace('_', ' ')}")
        plt.savefig(f'C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Histogram_of_{x}.png')
        #plt.show()
        plt.close()

# Calling the plot_hist function on the iris data set.
plot_hist(iris)
```
</details>


|||
|---|---|
|![Histogram of Sepal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_length.png)|![Histogram of Sepal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_width.png)|
|![Histogram of Petal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_length.png)|![Histogram of Petal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_width.png)|



__Discussion of histogram__

Reference to seaborn histogram, take from penguins file
What does the histogram tell us? Shape of histogram, distribution. 
Add comparison to literature. 


## Scatter plot of each pair of variables

The purpose of a scatter plot is to demonstrate the relationship between two variables. 

<details>
<summary>Scatter plot code</summary>

```python
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

# Call the plot_scatter function on the iris data set.
plot_scatter(iris)

# Use a pair plot! Much simplier method to generate a scatter plot of each pair of variables
g = sns.pairplot(iris, hue = 'species')
g.fig.suptitle('Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Pair_plot.png')
plt.close
#plt.show()
```

</details>
<br>

Reference to seaborn plot. 
COMMENT ON SCATTER PLOT


![Pair plot](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Pair_plot.png)

#### Any Other Analysis ####

## Correlation 


__Adding a line for best fit for Selected Variables__

Now that the data has been visualised with a scatter plot adding a line of best fit (also known as a regression line or a trend line) is useful to check if there is a linear or a non-linear relationship between the data points.

Matplotlib/Seaborn/Numpy analyses the data and fits a line that it thinks fits the data points the best. __HOW DOES IT DECIDE THAT? OLS, residuals__ The equation of a line is $y = mx + c$, where m is the slope and c is the y-intercept. 

```python
sepal_length_array = iris['sepal_length'].to_numpy()

sepal_width_array = iris['sepal_width'].to_numpy()

# Use numpy polyfit to fit a straight line between x and y.
# np.polyfit(x-axis, y-axis, deg). Deg = 1 for a linear equation.
m, c = np.polyfit(sepal_length_array, sepal_width_array, 1)

# Return values for the slope, m and y-intercept, c.
print(f'The value of the slope is {m.round(3)}.')
print(f'The value of the intercept is {c.round(3)}.')
```

    The value of the slope is -0.057.
    The value of the intercept is 3.389.


These values can then be used to plot the line. The y-values for the line are generated from the values of m and c above. 

<details>
<summary> Code to </summary>

```python
# Demonstrating how to plot a regression line on a scatter plot using numpy.
fig, ax = plt.subplots()

# A scatter plot of sepal Width vs sepal length using the numpy array generated in the previous cell.
ax.scatter(sepal_length_array, sepal_width_array)

# Plotting the trend line in green. The y-axis values are generated from the equation of the line, with m and c equal to the values generated above.
ax.plot(sepal_length_array, m * sepal_length_array + c, 'g-')

# Axis labels.
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Title.
plt.title('Sepal Width vs Sepal Length')
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Numpy_reg_plot.png')

```
</details>

![Numpy regression plot](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Numpy_reg_plot.png)

Fortunately are faster ways to add a regression line. Two of the simplest are seaborn's [regplot](https://seaborn.pydata.org/generated/seaborn.regplot.html) (regression plot) and [lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html) (linear model plot) functions. Regplot and lmplot generate very similiar plots but they have different parameters. Regplot is an axes-level function. Seaborn lmplot is a figure-level function with access to FacetGrid. FacetGrid means that multiple plots can be created in a grid with rows and columns. lmplot has the hue, col and row parameters __for categorical variables CHECK__. 

It is not possible to to extract the values of m and c from a seaborn plot. Linear regression  blah blah(https://medium.com/@shuv.sdr/simple-linear-regression-in-python-a0069b325bf8  

When creating plots it is very important to take any categorical variables into account. This will be demonstrated by creating some regression plots using regplot. To create side by side plots, regplot has the parameter of ax. The first plot will be a plot of the data and the second plot will take the flower species into account. 
* sepal width vs sepal length
* petal width vs petal length

<details>
<summary>Code for Regression Plot</summary>

```python
##### Regression Plots for Selected Variables

fig, ax = plt.subplots(2, 2, figsize = (15, 10))

# Regression plot between sepal length and sepal width
sns.regplot(iris, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 0])

# Regression plot between sepal width and sepal length by species
sns.regplot(setosa, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 1])
sns.regplot(versicolor, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 1])
sns.regplot(virginica, x = 'sepal_length', y = 'sepal_width', ax = ax[0, 1])

# Regression plot between petal length and petal width
sns.regplot(iris, x = 'petal_length', y = 'petal_width', ax = ax[1, 0])

# Regression plot between petal length and petal width by species
sns.regplot(setosa, x = 'petal_length', y = 'petal_width', ax = ax[1, 1])
sns.regplot(versicolor, x = 'petal_length', y = 'petal_width', ax = ax[1, 1])
sns.regplot(virginica, x = 'petal_length', y = 'petal_width', ax = ax[1, 1])

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
plt.show()
plt.close()

# Regression Line Pair Plot, kind = 'reg'
sns.pairplot(iris, hue = 'species', kind = 'reg')
plt.suptitle('Regression Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Pair_Regression_plots.png')

```
</details>


![Regression plots](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Regression_plots.png)

Simpson's paradox. Negative

![Pair regression plot](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Pair_Regression_plots.png)


## Calculate correlation coefficients

[Correlation](https://www.jmp.com/en_ca/statistics-knowledge-portal/what-is-correlation.html) tells us how two variables are related. The most commonly used method to calculate the value of the correlation coefficient, the Pearson method correlation is only suitable for linear plots so it is important to create a scatter plot of the variables before the correlation coefficient is calculated. The values of the correlation coefficient range from -1 to 1. The sign indicates the direction of the relationship, with -1 indicating a strong negative correlation (as x increases, y decreases), 0 indicates no correlation and +1 is a strong positive correlation (as x increases, y increases).

![Scatter plot and Correlation](https://files.realpython.com/media/py-corr-1.d13ed60a9b91.png)

_Image from realpython.com_

When the regression pair plot is analysed we can see that there is a linear relationship between all the variables in the data set so the correlation coefficient can be calculated using the [corr() function]() with the Pearson method. Correlation could also be carried out using numpy's np.corrcoeff().

``` python
# To calculate the correlation coefficient between two variables
corr_SL_vs_SW = iris['sepal_length'].corr(iris['sepal_width'])
print(f'The correlation coefficient between sepal length and sepal width is {corr_SL_vs_SW}')
```

Luckily there are a number of methods to calculate the correlation coefficent between all the numeric variables in one step. The first method uses the corr() and generates a correlation matrix. The second method involves creating a [seaborn heatmap](). A heatmap is a more visual method to display the same information as a correlation matrix. It is possible to create a [heatmap using matplotlib](https://www.geeksforgeeks.org/how-to-draw-2d-heatmap-using-matplotlib-in-python/) however it is not as simple as a seaborn heatmap.

<details>
<summary>Code for Correlation Matrix</summary>

```python
correlation_matrix = iris.drop(['species'], axis = 1).corr()
print(correlation_matrix)
```
</details>


|             |sepal_length |sepal_width |petal_length | petal_width|
|---|---|---|---|---|          
|sepal_length |    1.000000|   -0.109369|     0.871754|     0.81795|
|sepal_width  |   -0.109369|    1.000000|    -0.420516|    -0.35654|
|petal_length |    0.871754|   -0.420516|     1.000000|    0.962757|
|petal_width  |    0.817954|   -0.356544|     0.962757|     1.000000|

__Heatmap of correlation coefficients__

<details>
<summary> ADJUST TITLE Code to Calculate Correlation Coefficient using a Heatmap</summary>

```python
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
```

</details>



![Heat map](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Heatmap_correlation_coefficients.png)

The plot of sepal width vs sepal length is an example of Simpson's paradox. Wikipedia states that [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) is a phenomenon in probability and statistics in which a trend appears in several groups of data but disappears or reverses when the groups are combined. 


Who would have thought that nearly one hundred years later Anderson's data would be used by thousands of students worldwide who are learning statistics, data science or machine learning? https://www.sciencedirect.com/science/article/pii/S1877050919320836

## References

__Write output to a file in python__

In the command line use of file_name.py > outout_file_name.txt https://www.reddit.com/r/learnpython/comments/12emhsa/how_do_i_save_the_output_of_the_python_code_as_a/

Python File Write https://www.w3schools.com/python/python_file_write.asp

Writing to file in python https://www.geeksforgeeks.org/writing-to-file-in-python/


__Markdown references__

https://chrisfrew.in/blog/dropdowns-in-readmes/

__Plotting__

Figure-level vs axes-level functions https://seaborn.pydata.org/tutorial/function_overview.html#figure-level-vs-axes-level-functions

Python Seaborn Tutorial for Beginners: Start Visualizing Data https://www.datacamp.com/tutorial/seaborn-python-tutorial

Datacamp Introduction to Data Visualization with Matplotlib

Datacamp Introduction to Data Visualization with Seaborn

Understanding Boxplots https://builtin.com/data-science/boxplot

Countplot using seaborn in python https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/?ref=ml_lbp

Seaborn catplot https://www.geeksforgeeks.org/python-seaborn-catplot/?ref=lbp

__Correlation References__


