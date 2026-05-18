# DD2424 FINAL PROJECT README
Howdy y'all I put together this git repo for the final project. The readme will be pretty lean from the beginning, but add anything you think might be relevant to it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files

- `base_project.py` — shared dataset classes and training/evaluation utilities, imported by the notebook
- `basic_project.ipynb` — main notebook, works through the base assignment steps

## Dataset Setup

The dataset is too large to include in the repo. Download the Oxford-IIIT Pet Dataset following the directions in the assignment, then place it so your local directory looks like this:

```
final_project/
├── dataset/
│   └── oxford-iiit-pet/
│       ├── images.tar.gz
│       └── annotations.tar.gz
├── base_project.py
├── basic_project.ipynb
└── README.md
```

The notebook will extract the archives into `images/` and `annotations/` subdirectories on first run.

## TODO
| Task | Timeline | Responsible Person | Done |
|------|----------|--------------------|------|
| Setup pre trained net and dataset | 27/4 - 01/5 | Nicholas | ✅ |
| &nbsp;&nbsp;Sanity check with binary classification | | | ✅ |
| &nbsp;&nbsp;Multiclass classification | | Nicholas | ✅ | 
| &nbsp;&nbsp;Evaluate representation using linear probing | | |✅|
| Fine tune the last l layers | 01/5 - 06/5 | Thilini | ✅ |
| &nbsp;&nbsp;Gradually unfreezing layers during fine tune | | | ✅ |
| &nbsp;&nbsp;Evaluate model | | | ✅ |
| Fine tune with different fractions of data | 01/5 - 08/5 | Thilini | ✅ |
| Data augmentation | 01/5 - 10/5 | Chang | |
| &nbsp;&nbsp;L2 regularization | | Thilini | ✅ |
| &nbsp;&nbsp;Fine tune with imbalanced classes | | | |
| &nbsp;&nbsp;Evaluate model | | Thilini | ✅ |
| Reduced labelled data experiments | 08/5 - 13/5 | Thilini | ✅ |
| Prepare Semi-Supervised Setup (labelled and unlabeled data) | | | |
| &nbsp;&nbsp;Stratified labelled/unlabelled split | | | |
| &nbsp;&nbsp;LabelledDataset and UnlabelledDataset classes | | Thilini | ✅ |
| Define weak and strong augmentation | | | |
| &nbsp;&nbsp;Weak: flip + translation (Section 2.3) | | | |
| &nbsp;&nbsp;Strong: RandAugment magnitude=9 (Section 2.3) | | Thilini | ✅ |
| Implement FixMatch loss | | | |
| &nbsp;&nbsp;Supervised loss on weak augmented labelled (Eq. 3) | | | |
| &nbsp;&nbsp;Pseudo-labels from weak augmented unlabelled | | | |
| &nbsp;&nbsp;Confidence threshold mask τ=0.95 (Eq. 4) | | | |
| &nbsp;&nbsp;Unsupervised loss on strong augmented unlabelled | | | |
| &nbsp;&nbsp;Combined loss: loss_s + λu × loss_u | | Nicholas | |
| Training loop | | | |
| &nbsp;&nbsp;SGD + Nesterov + Cosine scheduler | | | |
| &nbsp;&nbsp;Mixed labelled + unlabelled batches per iteration | | | |
| &nbsp;&nbsp;Pseudo-labels recomputed every batch automatically — model improves → better pseudo-labels each iteration (Section 2.2) | | | |
| &nbsp;&nbsp;Save best model, plot training curves | 13/5 - 16/5 | Nicholas | |
| Evaluation | | | |
| &nbsp;&nbsp;Run supervised-only and FixMatch at fractions: 50%, 20%, 10%, 1% | | | |
| &nbsp;&nbsp;Test accuracy per fraction, Supervised vs FixMatch comparison, Save results per fraction | 13/5 - 18/5 | | |
| Ablations | | | |
| &nbsp;&nbsp;Confidence threshold, Augmentation magnitude, λu and µ, Pseudo-label recomputation frequency | 18/5 - 22/5 | | |
| Final Analysis and demo | | | |
| Report | 22/5 - 25/5 | | |