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
### BASE ASSIGNMENT
[x] setup pretrained net and dataset
    [x] download and install convnet
    [x] download dataset
[x] sanity check
[] multiclass classification
    [x] evaluate representation using linear probing