import matplotlib.pyplot as plt
import numpy as np

def plot_retrieval_comparison(without_reranker: dict, with_reranker: dict, save_path: str | None = None):
    """
    Create a bar plot comparing MAP and Recall between evaluations with and without a reranker.

    Args:
    without_reranker (dict): Evaluation results without reranker
    with_reranker (dict): Evaluation results with reranker
    save_path (str, optional): Path to save the plot. If None, the plot will be displayed instead.

    Returns:
    None
    """
    metrics = ['mean_average_precision', 'mean_recall']
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    without_values = [without_reranker[m] for m in metrics]
    with_values = [with_reranker[m] for m in metrics]

    rects1 = ax.bar(x - width/2, without_values, width, label='Without Reranker', color='skyblue')
    rects2 = ax.bar(x + width/2, with_values, width, label='With Reranker', color='orange')

    ax.set_ylabel('Score')
    ax.set_title('Comparison of Retrieval Metrics With and Without Reranker')
    ax.set_xticks(x)
    ax.set_xticklabels(['MAP', 'Recall'])
    ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
