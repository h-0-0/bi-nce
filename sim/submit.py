import slune

if  __name__ == "__main__":
    to_search_mle = {
        "n_train": [4000, 16000],
        "est": ['mle'],
        "K" : [10],
    }
    to_search_task_nce = {
        "n_train": [4000, 16000],
        "est": ['task_nce'],
        "K" : [16, 32, 128, 1024],
    }
    grid_mle = slune.searchers.SearcherGrid(to_search_mle)
    grid_task_nce = slune.searchers.SearcherGrid(to_search_task_nce) 
    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_mle)
    slune.sbatchit(script_path, template_path, grid_task_nce)