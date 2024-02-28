from polos.metrics.regression_metrics import RegressionReport
from polos.models import load_checkpoint
from tqdm import tqdm
from polos.models import download_model, load_checkpoint, model2download, str2model
from polos.trainer import TrainerConfig, build_trainer
import yaml
from utils import *
from dataset import *
from pascal50s import Pascal50sDataset
from PIL import Image

def collect_acc(memory, dataset_name, method, acc):
    memory.setdefault(dataset_name, {})
    memory[dataset_name].update({method : acc})
    gprint(f"[{dataset_name}]",method,acc)

def polos(dataset,args):
    yprint("Compute Polos ...")
    rep = RegressionReport()
    if args.model:
        model = load_checkpoint(args.model)
    elif args.hparams:
        yaml_file = yaml.load(open(args.hparams).read(), Loader=yaml.FullLoader)
        train_configs = TrainerConfig(yaml_file)
        model_config = str2model[train_configs.model].ModelConfig(yaml_file)
        print(str2model[train_configs.model].ModelConfig)
        print(model_config.namespace()) 
        model = str2model[train_configs.model](model_config.namespace())
        model.eval()
        model.freeze()
        
    data = []
    gt_scores = []
    for data_ in (pbar := tqdm(dataset)):
        pbar.set_description("Prepare dataset ...")
        data.append(data_)
    
    _, sys_score = model.predict(data,cuda=True,batch_size=32)
    return sys_score

def compute_acc(model_fn,dataset,**kwargs):
    data = {}
    gt = {}
    for (img_path, a, b, references, category_str, label) in (pbar := tqdm(dataset)):
        pbar.set_description("Prepare dataset ...")
        data.setdefault(category_str, {"A" : [], "B" : [], "gt": []})
        data[category_str]["A"].append({
            "img" : Image.open(img_path).convert("RGB"),
            "imgid" : img_path,
            "refs": references,
            "mt": a,
        })
        data[category_str]["B"].append({
            "img" : Image.open(img_path).convert("RGB"),
            "imgid" : img_path,
            "refs": references,
            "mt": b,
        })
        data[category_str]["gt"].append(label) # 0 if A > B else 1
    
    accs = {}
    for category_str, data_ in (pbar := tqdm(data.items())):
        pbar.set_description(f"Compute {category_str}")
        print("Compute type A ...")
        sys_scoreA = model_fn(data_["A"],**kwargs)
        
        print("Compute type B ...")
        sys_scoreB = model_fn(data_["B"],**kwargs)

        print("Compute accuracy ...")
        assert len(sys_scoreA) == len(sys_scoreB) == len(data_["gt"])
        acc, N = 0, len(sys_scoreA)
        for a, b, gt in zip(sys_scoreA,sys_scoreB,data_["gt"]):
            score = 0 if a > b else 1
            acc += 1 if score == gt else 0
        
        acc /= N
        accs[category_str] = acc
        rprint(f"acc({category_str}) : {acc}")
    
    return accs


def compute_pascal50S(args, memory, tops):
    dataset = Pascal50sDataset(root="data_en/pascal/", voc_path="data_en/pascal/VOCdevkit/VOC2010")
    dataset_name = "pascal50s"
    if args.polos:
        polos_acc = compute_acc(polos, dataset, args=args)
        collect_acc(memory, dataset_name, "Polos", polos_acc)

    # aggregate
    max_acc = {}
    for method, accs in memory[dataset_name].items():
        for category, acc in accs.items():
            max_acc.setdefault(category, ("",0))
            if max_acc[category][1] < acc:
                max_acc[category] = (method, acc)
    
    rprint("[TOP]")
    rprint(max_acc)
    tops[dataset_name] = max_acc

    return memory, tops