# pands-project

This is my analysis of the Fisher's Iris data set for the programming and scripting module.

## Project Statement
* Research the data set and summarise it in a README.
* Download the data set and add it to my repository. TICK DONE
* Write a program called analysis.py: 
    1. outputs a summary of each variable to a single text file
    2. saves a histogram of each variable to png files
    3. outputs a scatter plot of each pair of variables
    4. any other analysis


## Files in this project
analysis.py
analysis.ipynb
iris.data
README.md

# Summary of the data set

## History of the Data Set
In 1928 Edgar Anderson published his paper entitled ['The Problem of Species in the Northern Blue Flags, _Iris versicolor_ and _Iris virginica_'](https://www.biodiversitylibrary.org/page/15997721). Anderson was a evolutionary biologist interested in answering two questions namely, what are species and how have they originated? Between 1923 and 1928 he and his team studied _Iris versicolor_, at a number of different sites from Ontario in Canada to Alabama in the United States, by measuring a number of different iris characteristics. Surprisingly his study found that there were actually two different iris species present, _Iris versicolor_ and _Iris virginia_ and that it was possible to differentiate between them by geographic location. 

The data set is also known as Fisher's Iris Data set after the statistician and biologist, Ronald Fisher. The data measurements for _Isis setosa_ and _Iris versicolor_ were collected by Anderson from the same colony of plants in the Gaspé Peninsula, Quebec in 1935. According to [Unwin and Kleinman](https://www.jstor.org/stable/4331526?seq=13) the _Iris virginica_ data samples were from Anderson's original research and were collected in Camden, Tennessee. Fisher analysed the data and in 1936 published his results in the Annals of Eugenics [The Use of Multiple Measurements in Taxonomic Problems](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x). He used a statistical method, linear discriminant analysis to attempt to distinguish the different iris species from each other. He found that _Iris setosa_ was easily distinguishable from the other two iris species using this method. 

## What's in the data set?

The data set is a small data set with 150 rows and five columns with each row corresponding to a different flower sample. There are three different iris species, _Iris setosa_, _Iris versicolor_ and _Iris virginica_ with 50 samples for each species. 


![Iris species](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)


Four measurements (variables) were taken for each flower examined:
* sepal length
* sepal width
* petal length
* petal width

https://www.fs.usda.gov/wildflowers/beauty/iris/images/flower/blueflagiris_flower_lg.jpg

Each iris has three true petals and three sepals. The three petals are upright and are also know as standards. Sepals are a modified leaf and are sometimes known as falls. This diagram from nicely illustrates the petals and the sepals. 

![Sepals and petals](https://www.fs.usda.gov/wildflowers/beauty/iris/images/flower/blueflagiris_flower_lg.jpg)

![iris](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YYiQed4kj_EZ2qfg_imDWA.png)

The data set is available at [UCI Maching Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). The data set is very widely used with over 700,000 views of the data set on the UCI website. Who would have thought that nearly one hundred years later Anderson's data would be used by thousands of students worldwide who are learning statistics, data science or machine learning? 

