import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("iris.data", header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Having a look at the data set, checking that it loaded. 
head = f'The first five rows of the data set are: \n {iris.head()}\n \n'

#with open('analysis.txt', 'wt') as f:
    #f.write(head)

# Add column names
#iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
#add_header = f'Checking that the column names are correct: \n {iris.head()}\n \n'

#with open('analysis.txt', 'a') as f:
#    f.write(add_header)

for col in iris:
    sns.set_palette("Set1")
    sns.histplot(x = col, data = iris, hue = 'species')
    plt.title(f"Histogram of {col.title().replace('_', ' ')}")
    plt.xlabel(f"{col.replace('_', ' ')}")
    #plt.savefig(f'C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Histogram_of_{col}.png')
    plt.show()
    plt.close()