import torch
import clip
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

def read_image(imgid):
    from pathlib import Path
    vanilla = Path(imgid)
    fixed = Path(f"data_en/images/{imgid}")
    assert not (vanilla.exists() == fixed.exists()) # 両者共に存在/不在だと困る

    path = vanilla if vanilla.exists() else fixed
    return Image.open(path).convert("RGB")

class CLIPScore():
    def __init__(self,device="cuda"):
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device

    def batchify(self, targets, batch_size):
        return [targets[i:i+batch_size] for i in range(0,len(targets),batch_size)]

    def __call__(self, mt_list, refs_list, img_list, no_ref=False):
        B = 32
        mt_list, refs_list, img_list = [self.batchify(x,B) for x in [mt_list,refs_list,img_list]]
        scores = []
        assert len(mt_list) == len(refs_list) == len(img_list)
        for mt, refs, imgs in (pbar:= tqdm(zip(mt_list,refs_list, img_list),total=len(mt_list))):
            pbar.set_description(f"CLIPScore (noref: {no_ref})")
            imgs = [read_image(imgid) for imgid in imgs]
            refs_token = []
            for ref_list in refs:
                refs_token.append([clip.tokenize("A photo depicts " + ref,truncate=True).to(self.device) for ref in ref_list])

            refs = [torch.cat(ref,dim=0) for ref in refs_token]
            mts = clip.tokenize(["A photo depicts " + x for x in mt],truncate=True).to(self.device)
            imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs],dim=0).to(self.device)

            imgs = self.clip.encode_image(imgs)
            mts = self.clip.encode_text(mts)
            cos = F.cosine_similarity(imgs, mts,eps=0)
            cos[cos < 0.] = 0.
            clip_score = 2.5 * cos

            if no_ref:
                scores.extend(clip_score.tolist())
                continue

            cos = F.cosine_similarity(imgs, mts,eps=0)
            cos[cos < 0.] = 0.
            clip_score2 = cos

            assert clip_score.shape == clip_score2.shape
            clip_score = 2.0 * clip_score * clip_score2 / (clip_score + clip_score2)

            if not no_ref:
                scores.extend(clip_score.tolist())
        
        return scores
