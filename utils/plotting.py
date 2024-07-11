import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_mean_recall(mean_recall_results : list[dict[str, float]]):
    # Prepare data for plotting

    # Create a DataFrame for Seaborn
    data_list = []
    for result in mean_recall_results:
        df = pd.DataFrame(list(result.items()), columns=['Key', 'Average Value'])
        data_list.append(df)

    # Set the style and context for the plot
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create the plot with a custom line color
    plt.figure(figsize=(12, 8))
    for i,df in enumerate(data_list):
        sns.lineplot(data=df, x=df['Key'].astype(float), y='Average Value', marker='o', color=i)
        # Annotate each point with its value
        for i, row in df.iterrows():
            plt.text(i, row['Average Value'] + 0.001, f'{row["Average Value"]:.4f}', 
                    ha='center', va='bottom', fontsize=12, color='black')

    # Title and labels
    plt.title("Average Values for Each Key", fontsize=16, weight='bold')
    plt.xlabel("Keys", fontsize=14, weight='bold')
    plt.ylabel("Average Value", fontsize=14, weight='bold')


    # Show the plot
    plt.show()