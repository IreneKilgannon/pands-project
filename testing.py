
import plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data set, add headings to the columns.
penguins = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")

plotting.plot_hist(penguins)