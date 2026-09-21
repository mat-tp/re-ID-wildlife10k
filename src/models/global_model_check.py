import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.models.global_models import GlobalModel, extract_features
from src.data_processing.dataloader import WildlifeReID10kDataset

def main():
    model = GlobalModel().eval()
    ds = WildlifeReID10kDataset("data/metadata.csv", root="data", transform=model.eval_transform())

    # real values
    print(ds.data["split"].value_counts())
    print(ds.data["dataset"].value_counts().tail(8)) # The smallest 8 datasets

    # for name, rows in ds.data.groupby("dataset"):
    # # extract features for this dataset
    #     torch.save({"feats": f, "labels": y, "split": splits}, f"features/L384_{name}.pt")

    NAME = "AerialCattle2017"
    sub = ds.data[ds.data["dataset"] == NAME]
    print(sub["split"].value_counts())

    def feats(rows, limit=300):
        idx = rows.index.to_numpy()[:limit]
        loader = DataLoader(Subset(ds, idx), batch_size=8, num_workers=0)

        return extract_features(model, loader, device=None, amp=True)

    t = time.time()
    gal_f, (gal_y, _) = feats(sub[sub["split"] == "train"])
    qry_f, (qry_y, _) = feats(sub[sub["split"] == "test"])
    print(f"extracted {len(gal_f)} gallery + {len(qry_f)} query in {time.time()-t:.0f}s")

    pred = [gal_y[i] for i in (qry_f @ gal_f.T).argmax(1).tolist()]
    seen = np.mean([y in set(gal_y) for y in qry_y])
    print(f"top-1: {np.mean([p == y for p, y in zip(pred, qry_y)]):.3f}")
    print(f"fraction of query identities present in gallery: {seen:.3f}")

if __name__=="__main__":
    main()