# -*- coding: utf-8 -*-
import os

import click
import pandas as pd
import yaml

from torchnlp.download import download_file_maybe_extract

from .estimators import PolosEstimator, QualityEstimator
from .model_base import ModelBase
from .ranking import PolosRanker

str2model = {
    "PolosEstimator": PolosEstimator,
    "PolosRanker": PolosRanker,
    # Model that use source only:
    "QualityEstimator": QualityEstimator,
}

def get_cache_folder():
    if "HOME" in os.environ:
        cache_directory = os.environ["HOME"] + "/.cache/torch/yuigawada/"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
        return cache_directory
    else:
        raise Exception("HOME environment variable is not defined.")

def download_model(model: str, saving_directory: str = None) -> ModelBase:
    """Function that loads pretrained models from AWS.
    :param model: Name of the model to be loaded.
    :param saving_directory: RELATIVE path to the saving folder (must end with /).

    Return:
        - Pretrained model.
    """
    if saving_directory is None:
        saving_directory = get_cache_folder()

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
    
    if os.path.exists(saving_directory + "reprod/reprod.ckpt"):
        return saving_directory + "reprod/reprod.ckpt"
    
    models = {"polos" : "https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip"}

    if os.path.isdir(saving_directory + model):
        click.secho(f"{model} is already in cache.", fg="yellow")
        if not model.endswith("/"):
            model += "/"

    elif model not in models.keys():
        raise Exception(f"{model} is not a valid Polos model!")

    elif models[model].startswith("https://"):
        download_file_maybe_extract(models[model], directory=saving_directory)

    else:
        raise Exception("Something went wrong while dowloading the model!")

    if os.path.exists(saving_directory + model + ".zip"):
        os.remove(saving_directory + model + ".zip")

    click.secho("Download succeeded. Loading model...", fg="yellow")
    experiment_folder = saving_directory + "reprod"
    checkpoints = [
        file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
    ]
    checkpoint = checkpoints[-1]
    checkpoint_path = experiment_folder + "/" + checkpoint
    return checkpoint_path


def load_checkpoint(checkpoint: str) -> ModelBase:
    """Function that loads a model from a checkpoint file.
    :param checkpoint: Path to the checkpoint file.

    Returns:
        - Polos Model
    """
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    tags_csv_file = "/".join(checkpoint.split("/")[:-1] + ["meta_tags.csv"])
    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    if os.path.exists(tags_csv_file):
        # Uggly convertion from older Lightning checkpoints
        tags = pd.read_csv(
            tags_csv_file, header=None, index_col=0, squeeze=True
        ).to_dict()
        hparams = {}
        for k, v in tags.items():
            if isinstance(v, str) and v.replace(".", "", 1).isdigit():
                hparams[k] = float(v) if "." in v else int(v)
            else:
                hparams[k] = v
        model = str2model[tags["model"]].load_from_checkpoint(
            checkpoint, hparams=hparams
        )
    elif os.path.exists(hparam_yaml_file):
        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model = str2model[hparams["model"]].load_from_checkpoint(
            checkpoint, hparams=hparams
        )
    else:
        raise Exception(
            "[meta_tags.csv|hparams.yaml is missing from the checkpoint folder."
            " Please clean your cache folder (~/.cache/torch/yuigawada/) and try to download the model again."
        )

    model.eval()
    model.freeze()
    return model
