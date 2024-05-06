# pands-project 

Author: Irene Kilgannon

This is my analysis of the Fisher's Iris data set for the programming and scripting module.

## Making a plan
1. Using a readme for discussion  decided.
2. background research add some images
3. write summary statistics to .txt file - done. Discussion next
4. histograms - created a function, Create a module with all the plotting files?. Add discussion
5. scatter plot /pair plot between all the variables. Add discussion
6. perhaps some linear regression plots. Make a start today.
7. calculate correlation coefficients, fix labels on heatmap. add discussion
8. regression analysis https://campus.datacamp.com/courses/introduction-to-regression-with-statsmodels-in-python/simple-linear-regression-modeling?ex=1
9. machine learning 


Ideally I would create modules, with all the functions required for the file and import them. Having trouble with this. Probably classes would be better but I don't understand, know enough about them!

## Project Statement
* Research the data set and summarise it in a README.
* Download the data set and add it to my repository.
* Write a program called analysis.py: 
    1. outputs a summary of each variable to a single text file
    2. saves a histogram of each variable to png files
    3. outputs a scatter plot of each pair of variables
    4. any other analysis - suggested machine learning or regression analysis?? LEARN about them.


## Files in this project
analysis.py

analysis.ipynb

iris.data

README.md

plots directory with all the plots generated for this analysis

# Background to Fisher's Iris Data Set

In 1928 Edgar Anderson published his paper entitled ['The Problem of Species in the Northern Blue Flags, _Iris versicolor_ and _Iris virginica_'](https://www.biodiversitylibrary.org/page/15997721). Anderson was a evolutionary biologist interested in answering two questions namely, what are species and how have they originated? Between 1923 and 1928 he and his team studied _Iris versicolor_, at a number of different sites from Ontario in Canada to Alabama in the United States, by measuring a number of different iris characteristics. Surprisingly his study found that there were actually two different iris species present, _Iris versicolor_ and _Iris virginia_ and that it was possible to differentiate between them by geographic location. 

ADD IMAGE TO BREAK UP THE TEXT

The data set is commonly known as Fisher's Iris Data set after the statistician and biologist, Ronald Fisher. The data measurements for _Iris setosa_ and _Iris versicolor_ were collected by Anderson from the same colony of plants in the Gaspé Peninsula, Quebec in 1935. According to [Unwin and Kleinman](https://www.jstor.org/stable/4331526?seq=13) the _Iris virginica_ data samples were from Anderson's original research and were collected in Camden, Tennessee. Fisher collated and analysed the data and in 1936 published his results in the Annals of Eugenics [The Use of Multiple Measurements in Taxonomic Problems](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x). He used a statistical method, linear discriminant analysis to attempt to distinguish the different iris species from each other. He found that _Iris setosa_ was easily distinguishable from the other two iris species using this method. 

Fisher's data set can viewed in his published paper but, in our computer age, the data set is available to download at [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). The data set is very widely used with currently over 700,000 views of the data set on the UCI website.

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

A number of methods were explored to add the column names [adding headers to a dataframe in pandas a comprehensive-guide](https://saturncloud.io/blog/adding-headers-to-a-dataframe-in-pandas-a-comprehensive-guide/). The simplest method is to add the column names when the data set was loaded.

```python
iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
```

## Summary of the Data Set
***

[Analysis.txt](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.txt) contains the output of [analysis.py](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.py) and has provided the information used to summarise the data set.

[Python File Write](https://www.w3schools.com/python/python_file_write.asp)
[Writing to file in python](https://www.geeksforgeeks.org/writing-to-file-in-python/)

Checked that the file uploaded correctly and generated a txt file in one step.

Why did I save as a variable and not just add the code to the file? Wouldn't work otherwise. Add a reference to my inspiration. 

```python
shape = f'The shape of the data set is {iris.shape}. \n\n'

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
```

It is a small data set with 150 rows and five columns with each row corresponding to a different flower sample. There are three different iris species, _Iris setosa_, _Iris versicolor_ and _Iris virginica_ with 50 samples for each species. There is no missing data.

![iris](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YYiQed4kj_EZ2qfg_imDWA.png)

The five variables in the data set are:
* sepal_length (measured in cm)
* sepal_width (measured in cm)
* petal_length (measured in cm)
* petal_width (measured in cm)
* species

A basic description of the structure of an iris flower will help to understand the variable names. Each iris has three true petals and three sepals. The three petals are upright and are also known as standards. Sepals are a modified leaf and are usually green in colour and its function is to protect the developing flower bud. When the flower has bloomed the iris' sepal is described as "the landing pad for bumblebees" by the [US Forest Service](https://www.fs.usda.gov/wildflowers/beauty/iris/flower.shtml). This diagram nicely illustrates the difference between the petals and the sepals and also how the width and lenght of each were measured.

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


A [seaborn boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html) is nice visual method to display summary statistics about the data set. It gives us a lot of the same information that the describe method does but as it is visual it is to make comparisons.

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

plot_hist(iris)
```
</details>


|||
|---|---|
|![Histogram of Sepal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_length.png)|![Histogram of Sepal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_width.png)|
|![Histogram of Petal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_length.png)|![Histogram of Petal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_width.png)|


__Discussion of histogram__

Reference to seaborn histogram, take from penguins file
What does the histogram tell us?



## Scatter plot of each pair of variables

Code used. 
Reference to seaborn plot. 

![Scatter plot for each pair of variables](https://github.com/IreneKilgannon/pands-project/blob/main/Scatter_plot.png)

![Pair plot]()

#### Any Other Analysis ####

## Regression line

## Calculate correlation coefficients

Short paragraph about correlation.

Code used


|             |sepal_length |sepal_width |petal_length | petal_width|
| ---|---|---|  ---|---|          
|sepal_length |    1.000000|   -0.109369|     0.871754|     0.81795|
|sepal_width  |   -0.109369|    1.000000|    -0.420516|    -0.35654|
|petal_length |    0.871754|  -0.420516|    1.000000|    0.962757|
|petal_width  |    0.817954|  -0.356544|    0.962757|    1.000000|

Heat map of correlation coefficients

Fix labels on heat map


Who would have thought that nearly one hundred years later Anderson's data would be used by thousands of students worldwide who are learning statistics, data science or machine learning? https://www.sciencedirect.com/science/article/pii/S1877050919320836

## References

https://www.reddit.com/r/learnpython/comments/12emhsa/how_do_i_save_the_output_of_the_python_code_as_a/


https://chrisfrew.in/blog/dropdowns-in-readmes/

Plotting

Figure-level vs axes-level functions https://seaborn.pydata.org/tutorial/function_overview.html#figure-level-vs-axes-level-functions

Python Seaborn Tutorial for Beginners: Start Visualizing Data https://www.datacamp.com/tutorial/seaborn-python-tutorial

Datacamp Introduction to Data Visualization with Matplotlib

Datacamp Introduction to Data Visualization with Seaborn

Understanding Boxplots https://builtin.com/data-science/boxplot

Countplot using seaborn in python https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/?ref=ml_lbp

Seaborn catplot https://www.geeksforgeeks.org/python-seaborn-catplot/?ref=lbp