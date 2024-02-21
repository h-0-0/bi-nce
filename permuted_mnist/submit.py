import slune

if  __name__ == "__main__":
    to_search_bi = {
        'benchmark': ['permuted_mnist'],
        'model': ['MLP'], # FunnelMLP
        'learning_rate': [1e-4, 1e-6],
        'num_epochs': [150],
        'batch_size': [512],
        'est': ['bi_nce'],
        'patience': [5],
        'temperature': [0.01, 0.1],
        't_encoding': ['one_hot', 'repeated'],
    }
    grid_bi = slune.searchers.SearcherGrid(to_search_bi)

    to_search_info = {
        'benchmark': ['permuted_mnist'],
        'model': ['MLP'], # FunnelMLP
        'learning_rate': [1e-4, 1e-6],
        'num_epochs': [150],
        'batch_size': [512],
        'est': ['info_nce'],
        'patience': [5],
        'temperature': [0.01, 0.1],
        't_encoding': ['one_hot'],
    }
    grid_info = slune.searchers.SearcherGrid(to_search_info)
    
    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_bi)
    slune.sbatchit(script_path, template_path, grid_info)