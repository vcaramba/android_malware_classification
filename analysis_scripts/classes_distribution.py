import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_pie_chart(df, fre):
    labels = df[fre].astype('category').cat.categories.tolist()
    counts = df[fre].value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, radius=2000)
    ax1.axis('equal')
    plt.show()

def plot_all_classes_distribution(df):
    plt.figure(figsize=(16, 8))
    count = df['Label'].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(count.index, count.values, alpha=0.9)
    plt.title('Frequency Distribution', fontsize=20)
    plt.ylabel('Number of Occurrences', fontsize=15)
    plt.xlabel('Observation class (benign or malware family)', fontsize=15)
    plt.xticks(rotation=90)
    plt.show()


def plot_stacked_hist(df, x_var, groupby_var):
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]

    plt.figure(figsize=(16, 16), dpi=80)
    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

    plt.legend({group: col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by class", fontsize=8)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 100)
    plt.xticks(ticks=bins[::3], labels=[round(b, 1) for b in bins[::3]])
    plt.show()