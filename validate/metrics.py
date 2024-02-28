import json
import shutil
import os
from polos.metrics.regression_metrics import RegressionReport
from polos.models import load_checkpoint
from tqdm import tqdm
import json
from polos.models import download_model, load_checkpoint, model2download, str2model
from polos.trainer import TrainerConfig, build_trainer
import yaml
from utils import *
from dataset import *

def compute_polos_coef(args,test_dataset,dataset_name,kendall_type):
    yprint("Compute Polos ...")
    rep = RegressionReport(kendall_type)
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
    for data_ in (pbar := tqdm(test_dataset)):
        pbar.set_description("Prepare dataset ...")
        data.append(data_)
        gt_scores.append(data_["score"])
    
    _, sys_score = model.predict(data,cuda=True,batch_size=32)
    coef = rep.compute(sys_score, gt_scores)
    return coef