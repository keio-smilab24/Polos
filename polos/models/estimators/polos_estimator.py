# -*- coding: utf-8 -*-
import random
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch

from polos.models.estimators.estimator_base import Estimator
from polos.modules.feedforward import FeedForward
from polos.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors
import polos.clip as clip
import torch

from typing import List, Union

try:
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

class PolosEstimator(Estimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(Estimator.ModelConfig):
        switch_prob: float = 0.0

    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Estimator:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()

        if self.hparams.encoder_model != "LASER":
            self.layer = (
                int(self.hparams.layer)
                if self.hparams.layer != "mix"
                else self.hparams.layer
            )

            self.scalar_mix = (
                ScalarMixWithDropout(
                    mixture_size=self.encoder.num_layers,
                    dropout=self.hparams.scalar_mix_dropout,
                    do_layer_norm=True,
                )
                if self.layer == "mix" and self.hparams.pool != "default"
                else None
            )

        parallel_feature_extraction = True
        if parallel_feature_extraction:
            input_emb_sz = (
                self.encoder.output_units * 4 + 512*6
                if self.hparams.pool != "cls+avg"
                else self.encoder.output_units * 2 * 8
            )
        else:
            input_emb_sz = (
                self.encoder.output_units * 2 + 512*3
                if self.hparams.pool != "cls+avg"
                else self.encoder.output_units * 2 * 8
            )

        self.ff = torch.nn.Sequential(*[
            FeedForward(
                in_dim=input_emb_sz,
                # out_dim=input_emb_sz,
                hidden_sizes=self.hparams.hidden_sizes,
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
                final_activation=(
                    self.hparams.final_activation
                    if hasattr(
                        self.hparams, "final_activation"
                    )  # compatability with older checkpoints!
                    else "Sigmoid"
                ),
            ),
            torch.nn.Sigmoid()
        ])

        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.parallel_feature_extraction = parallel_feature_extraction


    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """ Sets different Learning rates for different parameter groups. """
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        ff_parameters = [
            {"params": self.ff.parameters(), "lr": self.hparams.learning_rate}
        ]

        if self.hparams.encoder_model != "LASER" and self.scalar_mix:
            scalar_mix_parameters = [
                {
                    "params": self.scalar_mix.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]

            optimizer = self._build_optimizer(
                layer_parameters + ff_parameters + scalar_mix_parameters
            )
        else:
            optimizer = self._build_optimizer(layer_parameters + ff_parameters)
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = collate_tensors(sample)
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = [self.encoder.prepare_sample(ref) for ref in sample["refs"]]
        
        inputs = {
            "mt_inputs": mt_inputs,
            "ref_inputs": ref_inputs,
            "refs": sample["refs"],
            "mt": sample["mt"],
            "imgs": sample["img"]
        }

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def masked_global_average_pooling(self, input_tensor, mask):
        mask = mask.logical_not() # mask[x] = input[x] is not pad
        mask_expanded = mask.unsqueeze(-1).expand_as(input_tensor).float()
        input_tensor_masked = input_tensor * mask_expanded
        num_elements = mask.sum(dim=1,keepdim=True).float() # TODO: チェック
        output_tensor = input_tensor_masked.sum(dim=1) / num_elements
        return output_tensor

    def forward(
        self,
        refs,
        mt,
        ref_inputs,
        mt_inputs,
        imgs: torch.tensor,
        alt_tokens: torch.tensor = None,
        alt_lengths: torch.tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        mt_tokens, mt_lengths = mt_inputs["tokens"], mt_inputs["lengths"]
        mt_sentemb, mt_sentembs, mt_mask, padding_index = self.get_sentence_embedding(mt_tokens, mt_lengths,pooling=False)
        mt_mask = mt_mask.logical_not()

        ref_sentemb_list = []
        ref_sentembs_list = []
        ref_mask_list = []
        for ref in ref_inputs:
            ref_tokens, ref_lengths = ref["tokens"], ref["lengths"]
            ref_sentemb, ref_sentembs, ref_mask, _ = self.get_sentence_embedding(ref_tokens, ref_lengths,pooling=False)
            ref_mask = ref_mask.logical_not()
            ref_sentemb_list.append(ref_sentemb)
            ref_sentembs_list.append(ref_sentembs)
            ref_mask_list.append(ref_mask)


        refs_clip = []
        for ref_list in refs: # (ref_cnt, B, L)
            subset = [clip.tokenize("A photo depicts " + ref,truncate=True).to(self.device) for ref in ref_list]
            subset = torch.cat(subset,dim=0)
            refs_tensor = self.clip.encode_text(subset)
            refs_clip.append(refs_tensor) 
        
        mts_clip = clip.tokenize(["A photo depicts " + x for x in mt],truncate=True).to(self.device)
        imgs_clip = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs],dim=0).to(self.device)

        imgs_clip = self.clip.encode_image(imgs_clip)
        mts_clip = self.clip.encode_text(mts_clip)
        del imgs

        scores = []
        for ref_sentemb, ref_clip in zip(ref_sentemb_list,refs_clip):
            diff = torch.abs(mt_sentemb - ref_sentemb)
            mul = mt_sentemb * ref_sentemb
            diff_clip = torch.abs(imgs_clip - mts_clip)
            mul_clip = imgs_clip * mts_clip
            diff_clip_txt = torch.abs(ref_clip - mts_clip)
            mul_clip_txt = ref_clip * mts_clip
            if self.parallel_feature_extraction:
                x = torch.cat(
                    (ref_sentemb,mt_sentemb,diff,mul,imgs_clip,mts_clip,diff_clip,mul_clip,diff_clip_txt,mul_clip_txt),dim=1
                )
            else:
                x = torch.cat(
                    (ref_sentemb,mt_sentemb,ref_clip,imgs_clip,mts_clip),dim=1
                )
            score = self.ff(x)
            scores.append(score)

        score = torch.max(torch.stack(scores),dim=0).values

        return {"score" : score}