import slune

if  __name__ == "__main__":
    to_search = {
        'benchmark': ['bi_mnist'],
        'model': ['MLP'], # FunnelMLP
        'learning_rate': [1e-1, 1e-3, 1e-5, 1e-7],
        'num_epochs': [100],
        'batch_size': [64, 256, 1024],
        'est': ['task_nce', 'info_nce'],
        'patience': [5],
        'temperature': [0.1, 0.5, 2.0],
    }
    grid = slune.searchers.SearcherGrid(to_search)
    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid)