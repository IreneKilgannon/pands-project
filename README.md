
<center><img src = "images\Fisher’s Iris Data Set.png" alt = "iris banner"></center>


Author: Irene Kilgannon

Student ID: G00220627

This is my analysis of the Fisher's Iris data set for the programming and scripting module for the [Higher Diploma in Science in Computing in Data Analytics](https://www.gmit.ie/higher-diploma-in-science-in-computing-in-data-analytics) at Atlantic Technological University.

## Left to do

NB add literature comparison, add more references where ever needed. 
1. Summary statistics. Improve Discussion
2. histograms. review
3. scatter plot. review
4. correlation coefficients, review 
5. regression analysis and machine learning https://campus.datacamp.com/courses/introduction-to-regression-with-statsmodels-in-python/simple-linear-regression-modeling?ex=1  add code comments, discussion, 

## Project Statement
* Research the data set and summarise it in a [README](https://github.com/IreneKilgannon/pands-project/blob/main/README.md).
* Download the data set and add it to my [GitHub repository](https://github.com/IreneKilgannon/pands-project).
* Write a program called [analysis.py](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.py): 
    1. outputs a summary of each variable to a single text file, [analysis.txt](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.txt)
    2. saves a histogram of each variable to png files.
    3. outputs a scatter plot of each pair of variables.
    4. any other analysis.

## Getting Started

__Step 1__ Download and install [Anaconda](https://www.anaconda.com/download). Anaconda is a Python distributon package and comes with pre-installed packages such as [pandas](https://pandas.pydata.org/docs/user_guide/index.html#user-guide), [NumPy](https://numpy.org/doc/stable/user/whatisnumpy.html), [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) which are required for this project. Please note that when installing Anaconda it is important to check the two boxes for:
  * Add Anaconda3 to my PATH environment variable
  * Register Anaconda3 as my default Python 3.9
  
![Anaconda]([Anaconda.png](https://github.com/IreneKilgannon/pands-project/blob/main/images/Anaconda.png))

__Step 2__ Download and install [Visual Studio Code](https://code.visualstudio.com/).

__Step 3__ Each file was written in VScode and pushed to [my GitHub account](https://github.com/IreneKilgannon) for assessment.

__Step 4__ To clone from GitHub enter the following in the command line:

    git clone https://github.com/IreneKilgannon/pands-project.git


## Get Help

If you have any questions or queries you can contact me at g00220627@atu.ie or alternatively [submit an issue](https://github.com/IreneKilgannon/pands-project/issues).


***

# Background to Fisher's Iris Data Set

In 1928 Edgar Anderson published his paper entitled ['The Problem of Species in the Northern Blue Flags, _Iris versicolor_ and _Iris virginica_'](https://www.biodiversitylibrary.org/page/15997721). Anderson was a evolutionary biologist interested in answering two questions namely, what are species and how have they originated? Between 1923 and 1928 he and his team studied _Iris versicolor_, at a number of different sites from Ontario in Canada to Alabama in the United States, by measuring a number of different iris characteristics. Surprisingly his study found that there were actually two iris species present, _Iris versicolor_ and _Iris virginia_ and that it was possible to differentiate between them by geographic location. This is reflected in the [common names of these two species of iris](https://hgic.clemson.edu/factsheet/rain-garden-plants-iris-versicolor-and-iris-virginica/). _Iris versicolor_ is commonly known as the Northern blue flag iris and _Iris virginica_ is commonly known as the Southern blue flag iris.

![iris](images//Iris_images.png)

The data set is commonly known as Fisher's Iris Data set after the statistician and biologist, [Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher). The data measurements for _Iris setosa_ and _Iris versicolor_ were collected by Anderson from the same colony of plants in the Gaspé Peninsula, Quebec in 1935. According to [Unwin and Kleinman](hhttps://academic.oup.com/jrssig/article/18/6/26/7038520) the _Iris virginica_ data samples were from Anderson's original research and were collected in Camden, Tennessee. Fisher collated and analysed the data and in 1936 published his results in the Annals of Eugenics [The Use of Multiple Measurements in Taxonomic Problems](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x). He used a statistical method, linear discriminant analysis to attempt to distinguish the different iris species from each other. He found that _Iris setosa_ was easily distinguishable from the other two iris species using this method. 

![Image from Fisher's paper](C:\Users\Martin\Desktop\pands\pands-project\images\fisher_data_set_image.png)

Fisher's data set can viewed in his published paper but, in our computer age, the data set is available to download at [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). The data set is still very widely used with currently over 700,000 views of the data set on the UCI website. It is worth mentioning that there is a move away from using this data set due to Fisher's work in eugenics.

## Import the Required Modules.

The following modules are required for this analysis:

* [pandas](https://www.w3schools.com/python/pandas/pandas_intro.asp) - for manipulating data and for performing data analysis.
* [numpy](https://www.w3schools.com/python/numpy/default.asp) - performs a wide variety of mathematical calculations on arrays.
* [matplotlib.pyplot](https://www.geeksforgeeks.org/python-introduction-matplotlib/) - used to create plots e.g. bar plots, scatter plots, histograms.
* [seaborn](https://realpython.com/python-seaborn/) - a python data visualisation library based on matplotlib. Usually requires less code syntax than matplotlib.
* [plotting](https://github.com/IreneKilgannon/pands-project/blob/main/plotting.py) - a module I developed that will create and save seaborn histograms and seaborn scatter plots for all the numeric data in a data set. __IF I CAN DEVELOP FURTHER TO PRINT FIG, AX PLOTS.__

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotting as pt
```

## Load the Data Set

The data set was downloaded from [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) and imported. This csv file does not contain column names and the column names were obtained from the variables table on the [information page of the iris data set](https://archive.ics.uci.edu/dataset/53/iris). The column names are sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm and species.

A number of methods were explored to add the column names [adding headers to a dataframe in pandas a comprehensive-guide](https://saturncloud.io/blog/adding-headers-to-a-dataframe-in-pandas-a-comprehensive-guide/). The quickest method is to add the column names using the name parameter when the data set is loaded. I did not add "_cm" to the columm names.

```python
iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
```

## Summary of the Data Set
***

_Task:_ Output a summary of each variable to a single text file, [analysis.txt](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.txt)

_Background_

```python
    with open('analysis.txt', 'wt') as f:
        f.write(head)
```

[With open()](https://ioflood.com/blog/python-open-file/#:~:text=Python-,Different%20File%20Modes,reading%20and%20writing%20text%20files) is used to open a file and the mode will then tell python what to do with the file. Two modes were used in this analysis. First write text, `'wt'` was used to both to create the analysis.txt file and to write to it. The append mode, `'a'` was used to append the results of the exploratory data analysis to analysis.txt. One major advantage of using with open() is that the file will automatically close.

`f.write()` will only write strings to a txt file. [This reddit post](https://www.reddit.com/r/learnpython/comments/12emhsa/how_do_i_save_the_output_of_the_python_code_as_a/) suggested saving the output to a variable and then result could be written to a txt file. 

<details>
<summary>Code written to analysis.txt</summary>

```python
# Investigating the variables to provide a summary for analysis.txt

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

# Append the to the analysis.txt file created previously
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
```
</details>
<br>

_Discussion_

The iris data set is a small data set with 150 rows and five columns with each row corresponding to a different flower sample. There is no missing data.

The five variables in the data set are:
* sepal_length
* sepal_width 
* petal_length
* petal_width 
* species 

There are three iris species, _Iris setosa_, _Iris versicolor_ and _Iris virginica_ in the species column with 50 samples for each species. The data type of the species column is object. 

The sepal length, sepal width, petal length and petal width columns are continuous numeric data of data type float(floating point number), which means that they are decimal numbers. They were all measured in cm to one decimal place.

A basic description of the structure of an iris flower will help to understand what each of the variable names are. Each iris has three true petals and three sepals. The three petals are upright and are sometimes known as standards. Sepals are a modified leaf, usually green in colour and its function is to protect the developing flower bud. When the flower has bloomed the iris's sepal is described as "the landing pad for bumblebees" by the [US Forest Service](https://www.fs.usda.gov/wildflowers/beauty/iris/flower.shtml). This diagram nicely illustrates the difference between the petals and the sepals and also how the width and length of each were measured.

![Length vs Width](https://www.integratedots.com/wp-content/uploads/2019/06/iris_petal-sepal-e1560211020463.png)

Summary statistics for numeric data are easily obtained using the [describe method](https://www.w3schools.com/python/pandas/ref_df_describe.asp). It gives a count of each variable and some statistics such as the min, max, mean and std deviation for each variable.

```python
# Summary statistics for the overall data set
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

The summary statistics reveal that petal lengths and petal widths are smaller than the sepal length and sepal width. Petal width is the smallest variable with the min length of 0.1 cm, a mean of 1.20 cm with max size of 2.5 mm. Sepal length is the largest of the four variables with a mean of 5.8 mm. I would imagine that the size difference between the sepal and petals would help an amateur botanist distinguish between the sepals and petals. __REPHRASE__


Summary Statistics for Each Iris Species

```python
# Summary statistics grouped by species. Transpose the result to make easier comparisons between the species.
summary_by_species = f"Summary statistics grouped by species \n{iris.groupby('species').describe().transpose()} \n\n"
```
Transposing the results of the summary statistics grouped by species gives a more readable form of the table. Without  transposing the results a table of 3 rows and 32 columns was created! The transposed table is much more readable and making direct comparisons is easier. 

COMMENT 

Summary statistics grouped by species
||||||
|---|--- |---|---|---|
|species      |      | Iris-setosa | Iris-versicolor  |Iris-virginica|
|sepal_length| count |   50.000000 |       50.000000  |     50.000000|
|            | mean  |    5.006000 |        5.936000  |      6.588000|
|            | std   |    0.352490 |        0.516171  |      0.635880|
|            | min   |    4.300000 |        4.900000  |      4.900000|
|            | 25%   |    4.800000 |        5.600000  |      6.225000|
|            | 50%   |    5.000000 |        5.900000  |      6.500000|
|            | 75%   |    5.200000 |        6.300000  |      6.900000|
|            | max   |    5.800000 |        7.000000  |      7.900000|
|sepal_width | count |   50.000000 |       50.000000  |     50.000000|
|            | mean  |    3.418000 |        2.770000  |      2.974000|
|            | std   |    0.381024 |        0.313798  |      0.322497|
|            | min   |    2.300000 |        2.000000  |      2.200000|
|            | 25%   |    3.125000 |        2.525000  |      2.800000|
|            | 50%   |    3.400000 |        2.800000  |      3.000000|
|            | 75%   |    3.675000 |        3.000000  |      3.175000|
|            | max   |    4.400000 |        3.400000  |      3.800000|
|petal_length| count |   50.000000 |       50.000000  |     50.000000|
|            | mean  |    1.464000 |        4.260000  |      5.552000|
|            | std   |    0.173511 |        0.469911  |      0.551895|
|            | min   |    1.000000 |        3.000000  |      4.500000|
|            | 25%   |    1.400000 |        4.000000  |      5.100000|
|            | 50%   |    1.500000 |        4.350000  |      5.550000|
|            | 75%   |    1.575000 |        4.600000  |      5.875000|
|            | max   |    1.900000 |        5.100000  |      6.900000|
|petal_width | count |   50.000000 |       50.000000  |     50.000000|
|            | mean  |    0.244000 |        1.326000  |      2.026000|
|            | std   |    0.107210 |        0.197753  |      0.274650|
|            | min   |    0.100000 |        1.000000  |      1.400000|
|            | 25%   |    0.200000 |        1.200000  |      1.800000|
|            | 50%   |    0.200000 |        1.300000  |      2.000000|
|            | 75%   |    0.300000 |        1.500000  |      2.300000|
|            | max   |    0.600000 |        1.800000  |      2.500000|

# PROBABLY DELETE THIS NOW, not required.
Summary statistics for _Iris setosa_ are:

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


Summary statistics for _Iris versicolor_ are: 
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


Summary statistics for _Iris virginica_ are: 
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
plt.savefig('plots\\Box_plot.png')
plt.close()
```
</details>


![Boxplot](plots/Box_plot.png)

The box plot clearly demonstrate that the petal lengths and petal widths of _Iris setosa_ is distincly different from the other two iris species. The petal length and widths are much smaller than _Iris versicolor_ and _Iris virginica_. This will aid in classifying _Iris setosa_.

----

## Histogram of each variable saved to png files

Histograms are used to plot continouous numeric data. The four variables of sepal length, sepal width, petal length and petal width fulfill this criteria.

<details>
<summary>Histogram code</summary>

```python
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

plt.suptitle('Histogram of the Iris Data Set')

plt.savefig('plots\\Summary_Histogram.png')
```
</details>

![Overall Histogram](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Summary_Histogram.png)

The four plots have an usual shape. The histograms should have a [normal distribution](https://www.youtube.com/watch?v=rzFX5NWojp0) and only one plot, the histogram of sepal width looks like it approaches a normal distribution. 

![Normal Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Bellcurve.svg/320px-Bellcurve.svg.png)

_Normal Distribtuion_

It is worth investigating to see if the species of the flower is affecting the shape of the curve. Rather than create a plot for each of the variables by coding each of the variables seperately (as I did above), I wrote function to plot each of the variables in the data set by looping through any column whose datatype is either integer or float. The histograms were created using a [seaborn histplot](). Seaborn's hue parameter makes it very easy to differentiate the data by the categorical variable, species. Hue is an optional argument for the function. 

<details>
<summary>Histogram code</summary>

```python
def plot_hist(df, hue = None):
    '''Plot a seaborn histogram of all the numeric variables in a dataframe. Optional hue parameter for a categorical variable. 

    Parameters
    ----------
    df : dataframe
    hue : a categorical variable in the data set. Optional argument.
    
    Returns
    -------
    A saved histogram of the numeric variables in the data set as a png file.
    '''
    for x in df:
        # Histograms are for continuous numeric data of data type integer or float.
        if df[x].dtype == 'int' or df[x].dtype == 'float':
            # Create a seaborn histogram, hue parameter is very useful to differentiate by a categorical variable.
            sns.histplot(x = x, data = df, hue = hue)
            # Add title. Replacing '_' with a blank space.
            plt.title(f"Histogram of {x.title().replace('_', ' ')}")
            # Label x-axis
            plt.xlabel(f"{x.replace('_', ' ')}")
            plt.ylabel('Frequency')
            # Save the plots
            plt.savefig(f'plots\\Histogram_of_{x}.png')
            plt.close()

# Call the plot_hist function from the plotting module on the iris data set.
pt.plot_hist(iris, hue = 'species')
```
</details>

|||
|---|---|
|![Histogram of Sepal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_length.png)|![Histogram of Sepal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_width.png)|
|![Histogram of Petal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_length.png)|![Histogram of Petal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_width.png)|


__Discussion of histogram__

Now that the data has been classified by species we can see that the histogram better resembles a normal distribution for most of the histograms. As the data set only has 50 data points for each species, it would require many more data points to fully resemble a normal distribution as stated in the Central limit theorem. and the unusual shape of the previous histograms was due to the overlapping data points. 

The histogram for petal length and petal width for _Iris setosa_ is different to the other histograms as it is right skewed. It is also distinct cluster. This could be helpful for classification of the species. __PHRASE BETTER__ __add comparison to literature__


## Scatter plot of each pair of variables

The purpose of a scatter plot is to demonstrate the relationship between two variables. Scatter plots also indicate if there are any outlying points (outliers) away from the main data points that could disrupt accurate correlation.
These were created using [seaborn scatter plots](https://www.geeksforgeeks.org/scatterplot-using-seaborn-in-python/). It is also possible to create scatter plots with [matplotlib's plt.scatter function](https://www.w3schools.com/python/python_ml_scatterplot.asp)

<details>
<summary>Scatter plot code</summary>

```python
def plot_scatter(df, hue = None):
    '''A function to plot a seaborn scatter plot of each pair of numeric variables in a dataframe.

    Parameters
    ----------
    df : dataframe
    hue : a categorical variable in the data set. Optional parameter/argument CHECK CORRECT TERM.
    
    Returns
    -------
    Saved scatter plots between all the numeric variables in the data set as a png file.
    '''
    plotted_x = []
    for x in df:
        plotted_x.append(x)
        # Only want a scatter plot of the numeric variables
        if df[x].dtype == 'int' or df[x].dtype == 'float':
            for y in df:
                # Only numeric data types will be plotted
                if df[y].dtype == 'int' or df[y].dtype == 'float':
                    # Do not create a plot if x and y are the same or if x has been used to create a plot previously.
                    if x == y or y in plotted_x:
                        continue
                    else:
                        # Create a scatter plot
                        sns.scatterplot(data = df, x = x, y = y, hue = hue)

                        # Add title to plot, removing any underscores and capitalizing it. 
                        plt.title(f"Scatter plot of {y.title().replace('_', ' ')} vs {x.title().replace('_', ' ')}")

                        # Label x and y-axis
                        plt.xlabel(f"{x.title().replace('_', ' ')}")
                        plt.ylabel(f"{y.title().replace('_', ' ')}")

                        # Save a scatter plot for each pair of variables
                        plt.savefig(f"plots\\Scatterplot_{y.title().replace('_', ' ')}_vs_{x.title().replace('_', ' ')}.png")
                        
                        plt.close()
                        
# Call the plot_scatter function from the plotting module on the iris data set.
pt.plot_scatter(iris, hue = 'species')

# Use a pair plot! Much simplier method to generate a scatter plot of each pair of variables
g = sns.pairplot(iris, hue = 'species')
g.fig.suptitle('Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('plots\\Pair_plot.png')
plt.close
#plt.show()
```

</details>
<br>

|Scatter plots between all the variables in the Iris data set||
|---|---|
|![PLvsSW](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Scatterplot_Petal%20Length_vs_Sepal%20Width.png)|![PLvsSL](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Scatterplot_Petal%20Length_vs_Sepal%20Length.png)|
|![PWvsPL](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Scatterplot_Petal%20Width_vs_Petal%20Length.png)|![PWvsSL](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Scatterplot_Petal%20Width_vs_Sepal%20Length.png)|
|![PWvsSW](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Scatterplot_Petal%20Width_vs_Sepal%20Width.png)|![SWvsSL](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Scatterplot_Sepal%20Width_vs_Sepal%20Length.png)|

The scatter plots demonstrate clearly that _Iris setosa_ is a distict cluster. It does not overlap with with _Iris versicolor_ or _Iris virginica_ in any of the scatter plots. It is important to always cross check what appears to be an outlier in the scatter plot with the box plot. What I thought was an obvious outlier in the scatter plot for _Iris setosa_ with a sepal width at 2.3 cm, is in the box plot the minimium value in the sepal width range for _Iris setosa_!

The clusters of _Iris versicolor_ and _Iris virginica_ overlap in all the scatter plots. In the plot of sepal length vs sepal width there is a significant amount of overlap. The least amount of overlap appears to be in the scatterplots between petal length and petal width and between petal width and sepal width. One widely used technique for classification in machine learning is called [K-nearest neighbours(KNN)](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn). The K-nearest neighbours algorithm looks at the nearest data points to the data point of interest and decides which cluster the data point belongs to. Minimising the overlap of clusters will improve the chance of correct cluster assignment.


![Pair plot](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Pair_plot.png)


#### Any Other Analysis ####

## Correlation Analysis

__What is Correlation?__

[Correlation](https://www.jmp.com/en_ca/statistics-knowledge-portal/what-is-correlation.html) tells us how two variables are related. It tells us what happens (if anything) to y if x increases. The value of the correlation coefficient ranges from -1 to 1. The sign indicates the direction of the relationship, with -1 indicating a strong negative correlation (as x increases, y decreases), 0 indicates no correlation and +1 is a strong positive correlation (as x increases, y increases).


![Scatter plot and Correlation](https://files.realpython.com/media/py-corr-1.d13ed60a9b91.png)

_Image from realpython.com_


__Why is correlation important?__

When choosing the variables to use for predictive modelling and machine learning it is important to understand how the variables interact together to decide what features are important. The value of the correlation coefficient gives guidance on the best variables to use as highly correlated variables will give more accurate models than poorly correlated variables. 

It is important to note that correlation does not mean causation. This means even though x and y are correlated, x does not necessarily cause y. There could be other variables, called [confounding variables](https://www.sciencedirect.com/topics/nursing-and-health-professions/confounding-variable) involved which are related to the variables of interest which could lead to misleading results. 

__Adding a line for best fit__

Before the correlation coefficient is calculated it is important to create a scatter plot with a line of best fit (also known as a regression line or a trend line) to check if there is a linear or a non-linear relationship between the data points. The most commonly used method to calculate the correlation coefficient, the [Pearson method correlation](https://www.youtube.com/watch?v=k7IctLRiZmo) is only suitable for linear plots. The spread of the data around the regression line indicates if there is good correlation between the variables. The data points for variables with a high correlation coefficient will be closer to the trend line. 

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

Fortunately are faster ways to add a regression line. Two of the simplest are seaborn's [regplot](https://seaborn.pydata.org/generated/seaborn.regplot.html) (regression plot) and [lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html) (linear model plot) functions. Regplot and lmplot generate very similiar plots but they have different parameters. Regplot is an axes-level function. Seaborn lmplot is a figure-level function with access to FacetGrid. FacetGrid means that multiple plots can be created in a grid with rows and columns. lmplot has the hue, col and row parameters __for categorical variables CHECK__. It is also possible to use the pair plot function with the kind parameter equal to reg to create a plot of all the numeric variables.

It is not possible to to extract the values of m and c from a seaborn plot. [Linear regression analysis](https://medium.com/@shuv.sdr/simple-linear-regression-in-python-a0069b325bf8 ) is required.


<details>
<summary>Regplot and lmplot</summary>

```python
# lmplot example. Sepal Width vs Sepal Length
sns.lmplot(iris, x = 'sepal_length_cm', y = 'sepal_width_cm', col = 'species')
plt.suptitle('Sepal Width vs Sepal Length by Species', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\lmplot_example.png')

# Regression Line Pair Plot, kind = 'reg'
sns.pairplot(iris, hue = 'species', kind = 'reg')
plt.suptitle('Regression Pair Plot of the Numeric Variables in the Iris Data Set', y = 1.05)
plt.savefig('C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Pair_Regression_plots.png')
```
</details>

![lmplot example, hue = species]()


![Pair regression plot](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Pair_Regression_plots.png)


## Calculate correlation coefficients

When the regression pairplot is analysed we can see that there is a linear relationship between all the variables in the data set so the correlation coefficient can be calculated using the [corr() function]() with the Pearson method. Correlation could also be carried out using numpy's np.corrcoeff().

``` python
# To calculate the correlation coefficient between two variables using numpy's corrcoeff()
corr_SL_vs_SW = iris['sepal_length'].corr(iris['sepal_width'])
print(f'The correlation coefficient between sepal length and sepal width is {corr_SL_vs_SW.round(3)}')
```

Once again there are a number of methods to calculate the correlation coefficent between all the numeric variables in one step. The first method uses the corr() and generates a [correlation matrix](https://datatofish.com/correlation-matrix-pandas/). The second method involves creating a [seaborn heatmap](). A heatmap is a more visual method to display the same information as a correlation matrix. It is possible to create a [heatmap using matplotlib](https://www.geeksforgeeks.org/how-to-draw-2d-heatmap-using-matplotlib-in-python/) however it is not as straightforward as a seaborn heatmap.

<details>
<summary>Code for Correlation Matrix</summary>

```python
correlation_matrix = iris.drop(['species'], axis = 1).corr()
print(correlation_matrix)
```
</details>

Correlation Matrix for the Iris Data Set.

|             |sepal_length |sepal_width |petal_length | petal_width|
|---|---|---|---|---|          
|sepal_length |    1.000000|   -0.109369|     0.871754|     0.81795|
|sepal_width  |   -0.109369|    1.000000|    -0.420516|    -0.35654|
|petal_length |    0.871754|   -0.420516|     1.000000|    0.962757|
|petal_width  |    0.817954|   -0.356544|     0.962757|     1.000000|


Petal length and petal width show a high positive correlation coefficient with a value of 0.963.

Both petal width and petal length have a high positive correlation with sepal length with values of 0.871 and 0.818 respectively. **Considering that the function of the sepal is to protect the developing flower bud it is not overly surprising that the sepal length has a high correlation with petal length. THINK ABOUT THIS**

Sepal width has a weak negative correlation with all the other variables in the data set with coefficient values ranging from -0.420 to -0.109.


__Heatmap of correlation coefficients__

In addition to creating a heatmap between all the variables in the data set, I will create individual heatmaps for each species. 

<details>
<summary> Code to Create a Heatmap of the Correlation Coefficients</summary>

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

These heatmaps demonstrate the importance of taking the categorical variables into account. For example the overall correlation coefficient between petal width and petal width was 0.96. When the coefficients of the indiviual species is taken into account the values range from 0.31 for _Iris setosa_, 0.79 for _Iris versicolor_ and 0.32 for _Iris virginica_. Another interesting pairing is petal width and petal length. 

This will be demonstrated by creating some regression plots using regplot. To create side by side plots, regplot has the parameter of ax. The first plot will be a plot of the overall data and the second plot will take the flower species into account. 
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

ADD COMMENT ON DIFFERENCES

The plot of sepal width vs sepal length is an example of Simpson's paradox. Wikipedia states that [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) is a phenomenon in probability and statistics in which a trend appears in several groups of data but disappears or reverses when the groups are combined. 

__COmparison with other's work__

https://zion-oladiran.medium.com/exploratory-data-analysis-iris-dataset-68897497b120

https://www.kaggle.com/code/danalexandru/simple-analysis-of-iris-dataset

https://www.hackersrealm.net/post/iris-dataset-analysis-using-python

https://www.angela1c.com/projects/iris_project/investigating-the-iris-dataset/


## Conclusion

Who would have thought that nearly one hundred years later Anderson's data would be used by thousands of students worldwide who are learning statistics, data science or machine learning? https://www.sciencedirect.com/science/article/pii/S1877050919320836


https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/

## References

www.canva.com

https://blog.hubspot.com/website/center-an-image-in-html#why-center-images-in-html-and-css

pandas transpose()  https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html

__Fisher and racism__

What is wrong with Ronald Aylmer Fisher https://profjoecain.net/what-is-wrong-ronald-aylmer-fisher/

The outstanding scientist, R.A. Fisher: his views on eugenics and race https://www.nature.com/articles/s41437-020-00394-6

RA Fisher and the science of hatred https://www.newstatesman.com/long-reads/2020/07/ra-fisher-and-science-hatred


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

Pearson's correlation, clearly explained https://www.youtube.com/watch?v=xZ_z8KWkhXE

r-squared https://statisticsbyjim.com/regression/interpret-r-squared-regression/

__Machine Learning__

https://www.kaggle.com/code/ash316/ml-from-scratch-with-iris



__ Literature search Recent uses of the iris data set__ 

https://www.sciencedirect.com/science/article/pii/S1877050919320836

https://ijece.iaescore.com/index.php/IJECE/article/view/33889

https://iopscience.iop.org/article/10.1088/1742-6596/2068/1/012004/pdf

https://peaceadegbite1.medium.com/iris-flower-classification-60790e9718a1#:~:text=The%20figures%20above%20show%20that,not%20greater%20than%203.8%20cm.

__name__ https://www.youtube.com/watch?v=sugvnHA7ElY&t=357s

modules references


/https://www.youtube.com/watch?v=CqvZ3vGoGs0&t=965s


***
END

