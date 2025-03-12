import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('medical_examination.csv')

# Calculate the 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Debugging: Print the aggregated DataFrame
    print("Aggregated DataFrame for categorical plot:")
    print(df_cat)

    # Debugging: Print the shape of df_cat
    print(f"Shape of df_cat: {df_cat.shape}")

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=4, aspect=1)

    # Set the y-axis label
    for ax in fig.axes.flat:
        ax.set_ylabel('total')  # Set the y-axis label to 'total'
        ax.set_title(f'Cardio: {ax.get_title().split("=")[-1].strip()}')  # Set title for clarity

    # Debugging: Print the number of bars in each subplot
    for ax in fig.axes.flat:
        num_bars = len(ax.patches)  # Count all patches (bars)
        print(f"Number of bars in subplot {ax.get_title()}: {num_bars}")

    # Save the figure
    fig.savefig('catplot.png')

    return fig.fig  # Return the figure from the FacetGrid

def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                  (df['height'] >= df['height'].quantile(0.025)) &
                  (df['height'] <= df['height'].quantile(0.975)) &
                  (df['weight'] >= df['weight'].quantile(0.025)) &
                  (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=ax)

    # Save the figure
    fig.savefig('heatmap.png')

    return fig
    
