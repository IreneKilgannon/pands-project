import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def exploratory(df):
    head = f"The first five rows of the data set are: \n {df.head()}\n \n"

    shape = f"The shape of the data set is {df.shape}. \n\n"

    # Get the variable names.
    column_names = f'Summary of the variable names in the data set are: \n {df.columns} \n\n'

    # Get the data types of the variables.
    data_types = f'The data types in the data set are: \n{df.dtypes}\n \n'

    # Look for missing data, NaN
    missing_values = f'Checking to see if there is any missing data or NaN. \n{df.isna().sum()} \n \n'

    # Uniques names in the species column.
    unique = f"The unique names in the species column are: \n {df['column'].unique()} \n\n"

    # Summary statistics for the overall the data set
    summary_statistics = f"Overall summary statistics for the data set: {df.describe()} \n\n"

    with open('analysis.txt', 'wt') as f:
        f.write(shape)
        f.write(column_names)
        f.write(data_types)
        f.write(missing_values)
        f.write(unique)
        f.write(summary_statistics)
    
        

def plot_hist(df, hue = None):
    '''To function plot a seaborn histogram of all the numeric variables in a dataframe.

    Parameters
    ----------
    df : dataframe
    hue : a categorical variable in the data set
    
    Returns
    -------
    A saved histogram of the numeric variables in the data set as a png file.
    '''
    for x in df:
        # Histograms are for continuous numeric data, continue to the next column if the datatype of the column is object.
        if df[x].dtype == 'int' or df[x].dtype == 'float':
        # Create a seaborn histogram, hue parameter is very useful to differentiate by another variable.
            sns.histplot(x = x, data = df, hue = hue)
            plt.title(f"Histogram of {x.title().replace('_', ' ')}")
            plt.xlabel(f"{x.replace('_', ' ')}")
            plt.savefig(f'Histogram_of_{x}.png')
            #plt.show()
            plt.close()

def plot_scatter(df, hue = None):
    '''To function plot a seaborn scatter plots of all the numeric variables in a dataframe.

    Parameters
    ----------
    df : dataframe
    hue : a categorical variable in the data set. Optional parameter/argument CHECK CORRECT TERM.
    
    Returns
    -------
    Saved scatter plots between all the numeric variables in the data set as a png file.
    '''
    # Initialize an empty list for plotted x columns
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
                        plt.savefig(f"Scatterplot_{y.title().replace('_', ' ')}_vs_{x.title().replace('_', ' ')}.png")

                        plt.close()

