import slune
import argparse

if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Wether to vary n_train, mx or mt', default='n_train')
    args = parser.parse_args()

    if args.exp == 'n_train':
        n_train = [2000, 4000, 8000, 16000, 32000, 64000]
        mx = [200]
        mt = [4]
    elif args.exp == 'mx':
        n_train = [16000]
        mx = [100, 200, 300, 400]
        mt = [4]
    elif args.exp == 'mt':
        n_train = [16000]
        mx = [200]
        mt = [2, 4, 8, 16]

    to_search_mle = {
        "n_train": n_train,
        "est": ['mle'],
        "K" : [10],
        "mx" : mx,
        "mt" : mt,
    }
    to_search_bi_nce = {
        "n_train": n_train,
        "est": ['bi_nce'],
        "K" : [4, 8, 16, 32],
        "mx" : mx,
        "mt" : mt,
    }
    grid_mle = slune.searchers.SearcherGrid(to_search_mle)
    grid_bi_nce = slune.searchers.SearcherGrid(to_search_bi_nce) 
    script_path = 'run_grid.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_mle)
    slune.sbatchit(script_path, template_path, grid_bi_nce)