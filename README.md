# pands-project 

Author: Irene Kilgannon

This is my analysis of the Fisher's Iris data set for the programming and scripting module.

## Making a plan
1. Using a readme
2. background research
3. write summary statistics to .txt file
4. histograms - created a function, Create a module with all the plotting files?
5. scatter plot /pair plot between all the variables
6. perhaps some linear regression plots
7. calculate correlation coefficients
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

## Import the Required Modules and Load the Data Set.

The data set was downloaded from [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) and imported. The data set did not contain any column names so these were added when the data set was imported. 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("iris_data.csv", names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
```

## Summary of the Data Set
***

[Analysis.txt](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.txt) contains the output of [analysis.py](https://github.com/IreneKilgannon/pands-project/blob/main/analysis.py) and has provided the information used to summarise the data set.

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

A basic description of the structure of an iris flower will help to understand the variable names. Each iris has three true petals and three sepals. The three petals are upright and are also known as standards. Sepals are a modified leaf and are usually green in colour and its function is to protect the developing flower bud. When the flower has bloomed the iris' sepal is described as "the landing pad for bumblebees" by the [US Forest Service](https://www.fs.usda.gov/wildflowers/beauty/iris/flower.shtml). This diagram from nicely illustrates the petals and the sepals. 

![Petals and sepals]()

This diagram illustrates the difference between the length and width measurements.

![Length vs Width](https://www.integratedots.com/wp-content/uploads/2019/06/iris_petal-sepal-e1560211020463.png)

 
## Summary Statistics for the data set


```python
# Summary statistics for the overall the data set
summary_statistics = f'Overall summary statistics for the data set. \n{iris.describe()} \n\n'

```
The describe method is very used to collate summary statistics about the data set. It gives a count of 

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

A box plot is another method to display summary statistics about the data set. It gives a visual comparison between all

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


|||
|---|---|
|![Histogram of Sepal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_length.png)|![Histogram of Sepal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_sepal_width.png)|
|![Histogram of Petal Length](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_length.png)|![Histogram of Petal Width](https://github.com/IreneKilgannon/pands-project/blob/main/plots/Histogram_of_petal_width.png)|



__Discussion of histogram__


## Scatter plot of each pair of variables

![Scatter plot for each pair of variables](https://github.com/IreneKilgannon/pands-project/blob/main/Scatter_plot.png)


|             |sepal_length |sepal_width |petal_length | petal_width|
| ---|---|---|  ---|---|          
|sepal_length |  |  1.000000|   -0.109369|     0.871754|     0.81795|
|sepal_width  |  | -0.109369|    1.000000|    -0.420516|    -0.35654|
|petal_length |    0.871754 |  -0.420516 |    1.000000 |    0.962757|
|petal_width  |    0.817954 |  -0.356544 |    0.962757 |    1.000000|



Who would have thought that nearly one hundred years later Anderson's data would be used by thousands of students worldwide who are learning statistics, data science or machine learning? https://www.sciencedirect.com/science/article/pii/S1877050919320836

## References

https://www.reddit.com/r/learnpython/comments/12emhsa/how_do_i_save_the_output_of_the_python_code_as_a/

