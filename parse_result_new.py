import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def create_heatmap(data, ax, n_features, vmin, vmax, xaxis, yaxis, i_context =False,  fontsize=20):
    pivot_data = data.pivot(index=yaxis, columns=xaxis, values='tv_dist')
    sns.heatmap(pivot_data, vmin = vmin, vmax=vmax, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f', cbar=False,
                annot_kws={'size': fontsize})  # Set font size for annotation
    itext = r'$\psi_{i, S}$' if i_context else r'$\psi_S$'
    ax.set_title(f'{itext}, n = {n_features}', fontsize=fontsize)
    axis_text = lambda t : 'Embed Dim' if t == 'embedding_dim' else '# Layers' if t == 'n_layers' \
                else '# Heads' if t == 'n_heads' else None

    ax.set_xlabel(axis_text(xaxis), fontsize=fontsize)
    ax.set_ylabel(axis_text(yaxis), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

def plot(xaxis, yaxis, fixed_values, filename):
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))

    xaxis_values = [2, 4, 16, 64] if xaxis == 'embedding_dim' else [1, 2, 4] if xaxis in ['n_layers', 'n_heads'] else None
    if xaxis_values is None:
        raise ValueError('Invalid xaxis value')
    
    n_features_values = [10, 25, 50, 100]

    # vmin_tvdist = result['tv_dist'].min()
    # vmax_tvdist = result['tv_dist'].max()
    for i, n_features in enumerate(n_features_values):
        nlayers = 2
        data = result[(result['n_features'] == n_features)]
        vmin_tvdist = data['tv_dist'].min()
        vmax_tvdist = data['tv_dist'].max()

        for k, v in fixed_values.items():
            data = data[data[k] == v]   

        ## for i in context
        data_1 = data[data['i_in_context'] == True]
        grouped_data = data_1.groupby([xaxis, yaxis]).mean(['tv_dist']).reset_index()

        create_heatmap(grouped_data, axes[i, 0], n_features, vmin_tvdist, vmax_tvdist, xaxis, yaxis, True)

        data_2 = data[data['i_in_context'] == False]
        
        grouped_data = data_2.groupby([xaxis, yaxis]).mean(['tv_dist']).reset_index()
        
        create_heatmap(grouped_data, axes[i, 1], n_features, vmin_tvdist, vmax_tvdist, xaxis, yaxis, False)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.savefig(f'{filename}.pdf')

if __name__=='__main__':
    result = pd.read_csv('results_all.csv')
    result = result[result['datapath']!='data/1']
    plot('embedding_dim', 'n_heads', dict(n_layers=1), 'plots_transformer/embedding_dim_vs_n_heads_n_layers_1')

    plot('embedding_dim', 'n_heads', dict(n_layers=2), 'plots_transformer/embedding_dim_vs_n_heads_n_layers_2')

    plot('embedding_dim', 'n_heads', dict(n_layers=4), 'plots_transformer/embedding_dim_vs_n_heads_n_layers_4')

    plot('embedding_dim', 'n_layers', dict(n_heads=1), 'plots_transformer/embedding_dim_vs_n_heads_n_heads_1')

    plot('embedding_dim', 'n_layers', dict(n_heads=2), 'plots_transformer/embedding_dim_vs_n_heads_n_heads_2')

    plot('embedding_dim', 'n_layers', dict(n_heads=4), 'plots_transformer/embedding_dim_vs_n_heads_n_heads_4')

    plot('n_heads', 'n_layers', dict(embedding_dim=2), 'plots_transformer/n_heads_vs_n_layers_embedding_dim_2')

    plot('n_heads', 'n_layers', dict(embedding_dim=4), 'plots_transformer/n_heads_vs_n_layers_embedding_dim_4')

    plot('n_heads', 'n_layers', dict(embedding_dim=16), 'plots_transformer/n_heads_vs_n_layers_embedding_dim_16')

    plot('n_heads', 'n_layers', dict(embedding_dim=64), 'plots_transformer/n_heads_vs_n_layers_embedding_dim_64')



