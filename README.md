# pands-project 

Author: Irene Kilgannon

This is my analysis of the Fisher's Iris data set for the programming and scripting module.

## Making a plan
1. decide if using readme or jupyter notebooks for comments
2. background research
3. write summary statistics to .txt file
4. histograms
5. scatter plot /pair plot between all the variables
6. perhaps some linear regression plots
7. calculate correlation coefficients
8. regression analysis https://campus.datacamp.com/courses/introduction-to-regression-with-statsmodels-in-python/simple-linear-regression-modeling?ex=1
9. machine learning

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

# Background to Fisher's Iris Data Set

In 1928 Edgar Anderson published his paper entitled ['The Problem of Species in the Northern Blue Flags, _Iris versicolor_ and _Iris virginica_'](https://www.biodiversitylibrary.org/page/15997721). Anderson was a evolutionary biologist interested in answering two questions namely, what are species and how have they originated? Between 1923 and 1928 he and his team studied _Iris versicolor_, at a number of different sites from Ontario in Canada to Alabama in the United States, by measuring a number of different iris characteristics. Surprisingly his study found that there were actually two different iris species present, _Iris versicolor_ and _Iris virginia_ and that it was possible to differentiate between them by geographic location. 

The data set is commonly known as Fisher's Iris Data set after the statistician and biologist, Ronald Fisher. The data measurements for _Isis setosa_ and _Iris versicolor_ were collected by Anderson from the same colony of plants in the Gaspé Peninsula, Quebec in 1935. According to [Unwin and Kleinman](https://www.jstor.org/stable/4331526?seq=13) the _Iris virginica_ data samples were from Anderson's original research and were collected in Camden, Tennessee. Fisher collated and analysed the data and in 1936 published his results in the Annals of Eugenics [The Use of Multiple Measurements in Taxonomic Problems](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x). He used a statistical method, linear discriminant analysis to attempt to distinguish the different iris species from each other. He found that _Iris setosa_ was easily distinguishable from the other two iris species using this method. 

Fisher's data set can be seen in his published paper but, in our computer age, the data set is available to download at [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). The data set is very widely used with currently over 700,000 views of the data set on the UCI website.

## Summary of the Data Set
***

Analysis.txt is the output of analysis.py provides the information I here to summarise the data set. IMPORVE

```python 
print(f'Summary of the variables and the data types in the data set.')
print(iris.info())
```


It is a small data set with 150 rows and five columns with each row corresponding to a different flower sample. There are three different iris species, _Iris setosa_, _Iris versicolor_ and _Iris virginica_ with 50 samples for each species. There is no data missing in any of the columns.

![iris](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YYiQed4kj_EZ2qfg_imDWA.png)

Four measurements (or variables) were taken for each flower examined:
* sepal length in cm
* sepal width in cm
* petal length in cm
* petal width in cm

Each iris has three true petals and three sepals. The three petals are upright and are also know as standards. Sepals are a modified leaf and are sometimes called falls. Sepals are usually green in colour and its function is to protect the developing flower bud. When the flower has bloomed the iris' sepal is described as "the landing pad for bumblebees" by the [US Forest Service](https://www.fs.usda.gov/wildflowers/beauty/iris/flower.shtml). This diagram from nicely illustrates the petals and the sepals. 

![Petals and sepals](https://www.fs.usda.gov/wildflowers/beauty/iris/images/flower/blueflagiris_flower_lg.jpg)

https://www.integratedots.com/wp-content/uploads/2019/06/iris_petal-sepal-e1560211020463.png

![Boxplot](c:/Users/Martin/Desktop/pands/pands-project/plots/Boxplot.png)

 Who would have thought that nearly one hundred years later Anderson's data would be used by thousands of students worldwide who are learning statistics, data science or machine learning? https://www.sciencedirect.com/science/article/pii/S1877050919320836


## Histogram of each variable saved to png files

![Sepal Width](https://github.com/IreneKilgannon/pands-project/blob/main/Sepal%20Width%20(cm).png)

![Sepal length](https://github.com/IreneKilgannon/pands-project/blob/main/Sepal%20Length%20(cm).png)

![Petal Length](https://github.com/IreneKilgannon/pands-project/blob/main/Petal%20Length%20(cm).png)

![Petal Width](https://github.com/IreneKilgannon/pands-project/blob/main/Petal%20Length%20(cm).png)


__Discussion of histogram__


## Scatter plot of each pair of variables

![Scatter plot for each pair of variables](https://github.com/IreneKilgannon/pands-project/blob/main/Scatter_plot.png)


## References

https://www.reddit.com/r/learnpython/comments/12emhsa/how_do_i_save_the_output_of_the_python_code_as_a/

