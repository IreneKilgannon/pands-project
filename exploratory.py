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
    
        

def plot_hist(df):
    for col in df:
        sns.histplot(x = col, data = df, hue = 'species')
        plt.title(f"Histogram of {col.title().replace('_', ' ')}")
        plt.xlabel(f"{col.replace('_', ' ')}")
        plt.savefig(f'Histogram_of_{col}.png')
        #plt.show()
        plt.close()

