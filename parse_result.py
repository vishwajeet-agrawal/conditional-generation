import matplotlib.pyplot as plt

import pandas as pd
def main_plot():
# Read the CSV data
    df = pd.read_csv('results.csv')
    df = df[df['step']==500]

    print(df.info())

    # Show the first few rows
    print(df.head())

    # Calculate average loss and tv_dist for each combination of embedding_dim, n_features, and caggregate
    grouped = df.groupby(['embedding_dim', 'n_features', 'caggregate']).agg({
        'tv_dist': 'mean'
    }).reset_index()

    print("\nAverage loss and tv_dist for each combination:")
    print(grouped)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('TV Distance vs Embedding Dimension for different n_variables', fontsize=30)
    axs = axs.flatten()

    # Plot for each n_features
    for i, n_features in enumerate(sorted(df['n_features'].unique())):
        data = grouped[grouped['n_features'] == n_features]
        
        for caggregate in data['caggregate'].unique():
            subset = data[data['caggregate'] == caggregate]
            axs[i].plot(subset['embedding_dim'], subset['tv_dist'], marker='o', label=f'{caggregate}')
    
        
        axs[i].text(3, 0.18, f'n_variables = {n_features}', fontsize = 30)
        # axs[i].set_title(f'n_features = {n_features}', fontsize = 30)
        axs[i].legend(fontsize = 30)
        axs[i].grid(True)
        axs[i].set_ylim(0, 0.2)
        axs[i].tick_params(axis='x', which='major', labelsize=30)
        axs[i].set_xlabel('Embedding Dimension', fontsize = 30)
        if i %2 == 0:
            axs[i].set_ylabel('Average TV Distance', fontsize = 30)
            axs[i].tick_params(axis='y', which='major', labelsize=30)
        if i %2 != 0:
            axs[i].set_yticklabels([])
        axs[i].set_xscale('log')
        axs[i].set_xticks([1, 2, 5, 10, 25, 50, 100, 300])
        axs[i].set_xticklabels([1, 2, 5, 10, 25, 50,100, 300])
        

    plt.tight_layout()
    plt.savefig('tv_dist_vs_embedding_dim.pdf')
    plt.close()
def main_table():
    df = pd.read_csv('results.csv')
    df = df[df['step']==500]

    # print(df.info())

    # Show the first few rows
    # print(df.head())
    grouped = df.groupby(['embedding_dim', 'n_features', 'caggregate']).agg({
        'tv_dist': 'mean'
    }).reset_index()
    df = df[['n_features', 'caggregate', 'embedding_dim', 'tv_dist']]
    df = df.pivot_table(index=['n_features', 'caggregate'], columns=['embedding_dim'], values=['tv_dist'])
    # pd.set_option('display.float_format', lambda x: '%.2f' % x)
    df.round(2).to_latex('tv_dist_table.tex')
if __name__ == '__main__':
    main_plot()