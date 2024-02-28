import json
import argparse
import json
from utils import *
from dataset import *
from metrics import *
from compute_correlations import compute_flickr
from compute_pascal50s import compute_pascal50S
from compute_foil import compute_foil

def collect_coef(memory, dataset_name, method, coef_tensor):
    memory.setdefault(dataset_name, {})
    coef = {k : round(float(v.numpy() if not isinstance(v,float) else v),4) for k, v in coef_tensor.items()}
    memory[dataset_name].update({method : coef})
    gprint(f"[{dataset_name}]",method,coef)

def compute_coef(args,memory,tops):
    dataset_name = "test"
    path = f"data_en/polaris/polaris_{dataset_name}.csv"
    yprint(f"Processing {dataset_name} ... (path: {path})")
    test_dataset = get_dataset(path)

    # mypolos
    if args.polos:
        polos_coef = compute_polos_coef(args,test_dataset,dataset_name,kendall_type='c')
        collect_coef(memory, dataset_name, "Polos", polos_coef)

    return memory, tops


def main(args):
    memory, tops = {}, {}
    if args.flickr:
        memory, tops = compute_flickr(args,args.model,memory,tops)
    if args.coef:
        memory, tops = compute_coef(args, memory, tops)
    if args.pascal:
        memory, tops = compute_pascal50S(args, memory, tops)
    if args.foil:
        memory, tops = compute_foil(args, memory, tops)

    with open("zeroshot_test_results.json", "w") as f:
        json.dump(memory, f, indent=4)
    
    yprint("[RESULTS]")
    gprint(json.dumps(memory, indent=4))

    rprint("[TOP]")
    for dataset_name, values in tops.items():
        rprint(f"> {dataset_name}")
        if isinstance(values,dict): # coef
            for kind, coef in values.items():
                rprint(f"{kind}: {coef[0]} ({coef[1]})")
        else: # acc
            method, acc = values
            rprint(f"{method} ({acc})")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # models
    parser.add_argument('--model', default=None)
    parser.add_argument('--hparams',default=None)
    parser.add_argument('--polos', action='store_true')

    # benchmarks
    parser.add_argument('--coef', action='store_true')
    parser.add_argument('--flickr', action='store_true')
    parser.add_argument('--pascal', action='store_true')
    parser.add_argument('--foil', action='store_true')
    
    args = parser.parse_args()
    main(args)
