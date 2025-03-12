import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Debugging: Print the aggregated DataFrame
    print("Aggregated DataFrame for categorical plot:")
    print(df_cat)

    # Debugging: Print the shape of df_cat
    print(f"Shape of df_cat: {df_cat.shape}")

    # 8
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='bar', height=4, aspect=1)

    # Set the y-axis label
    for ax in fig.axes.flat:
        ax.set_ylabel('total')  # Set the y-axis label to 'total'
        ax.set_title(f'Cardio: {ax.get_title().split("=")[-1].strip()}')  # Set title for clarity

    # Debugging: Print the number of bars in each subplot
    for ax in fig.axes.flat:
        num_bars = len(ax.patches)  # Count all patches (bars)
        print(f"Number of bars in subplot {ax.get_title()}: {num_bars}")

    # Debugging: Print the number of bars in the plot
    total_bars = sum(len(ax.patches) for ax in fig.axes.flat)  # Count patches in all axes
    print(f"Total number of bars in the plot: {total_bars}")

    # 9
    return fig.fig  # Return the figure from the FacetGrid
# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                  (df['height'] >= df['height'].quantile(0.025)) &
                  (df['height'] <= df['height'].quantile(0.975)) &
                  (df['weight'] >= df['weight'].quantile(0.025)) &
                  (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
