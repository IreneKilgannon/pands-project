

def plot_hist(df):
    for col in df:
        sns.histplot(x = col, data = df, hue = 'species')
        plt.title(f"Histogram of {col.title().replace('_', ' ')}")
        plt.xlabel(f"{col.replace('_', ' ')}")
        plt.savefig(f'C:\\Users\\Martin\\Desktop\\pands\\pands-project\\plots\\Histogram_of_{col}.png')
        #plt.show()
        plt.close()

