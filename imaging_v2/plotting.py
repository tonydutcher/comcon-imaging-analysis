import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context="paper", font="monospace")

def heatmap(dd, figsize=(14,12), zscore = False, vmin=-1,vmax=1, title=None):
    df = dd.copy()
    if zscore:
        for col in df.columns:
            df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    df.index = df.columns
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(df, linewidths=0.5, ax=ax, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticklabels(df.index[::-1], rotation=0)    
    return fig

def correlation_boxplot(df1, df2, figsize=(4,10)):
    #corr = df1.corrwith(df2.head(len(df1))[:len(df1)])
    import numpy as np
    corr = np.corrcoef(df1.values, df2.values[:len(df1),:len(df1)] )
    import pdb; pdb.set_trace()
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(corr, orient='v', ax=ax)
    return fig
