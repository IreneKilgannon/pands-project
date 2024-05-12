import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(df, hue = None):
    '''To function plot a seaborn histogram of all the numeric variables in a dataframe.

    Parameters
    ----------
    df : dataframe
    hue : a categorical variable in the data set
    
    Returns
    -------
    A saved histogram in the plots directory for the numeric variables in the data set as a png file.
    '''
    for x in df:
        # Histograms are for continuous numeric data, continue to the next column if the datatype of the column is object.
        if df[x].dtype == 'int' or df[x].dtype == 'float':
        # Create a seaborn histogram, hue parameter is very useful to differentiate by another variable.
            sns.histplot(x = x, data = df, hue = hue)
            plt.title(f"Histogram of {x.title().replace('_', ' ')}")
            plt.xlabel(f"{x.replace('_', ' ')}")
            plt.savefig(f'plots\\Histogram_of_{x}.png')
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
                        plt.savefig(f"plots\\Scatterplot_{y.title().replace('_', ' ')}_vs_{x.title().replace('_', ' ')}.png")

                        plt.close()

## DO i actually need this?
if __name__ == '__main__':
    plot_hist()
    plot_scatter()