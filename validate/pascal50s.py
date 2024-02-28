# from https://github.com/jmhessel/clipscore/issues/4

import torch
import random
import scipy
import os
from tqdm import tqdm

class Pascal50sDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root: str = "data/Pascal-50s/",
                 media_size: int = 224,
                 voc_path: str = "data/VOC2010/"):
        super().__init__()
        self.voc_path = voc_path
        self.fix_seed()
        self.read_data(root)
        self.read_score(root)
        self.idx2cat = {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}

    @staticmethod
    def loadmat(path):
        return scipy.io.loadmat(path)

    def fix_seed(self, seed=42):
        torch.manual_seed(seed)
        random.seed(seed)

    def read_data(self, root):
        mat = self.loadmat(
            os.path.join(root, "pyCIDErConsensus/pair_pascal.mat"))
        self.data = mat["new_input"][0]
        self.categories = mat["category"][0]
        # sanity check
        c = torch.Tensor(mat["new_data"])
        hc = (c.sum(dim=-1) == 12).int()
        hi = (c.sum(dim=-1) == 13).int()
        hm = ((c < 6).sum(dim=-1) == 1).int()
        mm = ((c < 6).sum(dim=-1) == 2).int()
        assert 1000 == hc.sum()
        assert 1000 == hi.sum()
        assert 1000 == hm.sum()
        assert 1000 == mm.sum()
        assert (hc + hi + hm + mm).sum() == self.categories.shape[0]
        chk = (torch.Tensor(self.categories) - hc - hi * 2 - hm * 3 - mm * 4)
        assert 0 == chk.abs().sum(), chk

    def read_score(self, root):
        mat = self.loadmat(
            os.path.join(root, "pyCIDErConsensus/consensus_pascal.mat"))
        data = mat["triplets"][0]
        self.labels = []
        self.references = []
        for i in range(len(self)):
            votes = {}
            refs = []
            for j in range(i * 48, (i + 1) * 48):
                a,b,c,d = [x[0][0] for x in data[j]]
                key = b[0].strip() if 1 == d else c[0].strip()
                refs.append(a[0].strip())
                votes[key] = votes.get(key, 0) + 1
            assert 2 >= len(votes.keys()), votes
            assert len(votes.keys()) > 0
            try:
                vote_a = votes.get(self.data[i][1][0].strip(), 0)
                vote_b = votes.get(self.data[i][2][0].strip(), 0)
            except KeyError:
                print("warning: data mismatch!")
                print(f"a: {self.data[i][1][0].strip()}")
                print(f"b: {self.data[i][2][0].strip()}")
                print(votes)
                exit()
            # Ties are broken randomly.
            label = 0 if vote_a > vote_b + random.random() - .5 else 1 # a == bの場合は0.5の確率で0か1を選ぶ
            self.labels.append(label)
            self.references.append(refs)

    def __len__(self):
        return len(self.data)

    def get_image_path(self, filename: str):
        path = os.path.join(self.voc_path, "JPEGImages")
        return os.path.join(path, filename)

    def __getitem__(self, idx: int):
        vid, a, b = [x[0] for x in self.data[idx]]
        label = self.labels[idx]
        img_path = self.get_image_path(vid)
        a = a.strip()
        b = b.strip()
        references = self.references[idx]
        category = self.categories[idx]
        category_str = self.idx2cat[category]
        return img_path, a, b, references, category_str, label

def sanity_check(detail=False):
    # sanity check
    dataset = Pascal50sDataset(root="pascal/", voc_path="pascal/VOCdevkit/VOC2010")
    one_sample = dataset[0]
    assert one_sample is not None

    dprint = lambda *args, **kwargs: print(*args, **kwargs) if detail else None
    for it, one_sample in enumerate(tqdm(dataset)):
        dprint("="*20)
        dprint("sample:",it)
        dprint("="*20)
        img_path, a, b, references, category, label = one_sample
        assert os.path.exists(img_path)
        
        dprint("img_path:", img_path)
        dprint("a:", a)
        dprint("b:", b)
        dprint("references:", references)
        dprint("category:", category)
        dprint("label:", label)

if __name__ == "__main__":
    sanity_check(detail=False)
