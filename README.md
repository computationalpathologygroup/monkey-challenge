# The MONKEY challenge: Machine-learning for Optimal detection of iNflammatory cells in KidnEY transplant biopsies
This repository contains all tutorials and code in connection with the MONKEY challenge run on [Grand Challenge](https://monkey.grand-challenge.org/)

## Baseline algorithm tutorial
The folder `tutorials` contains the code to get started with the MONKEY challenge.
The Jupyter notebooks show you how to preprocess the data, train a model and run inference.

## Creating the inference docker image
Following soon.

## Results evaluation
The folder `evaluation` contains the code to evaluate the results of the MONKEY challenge. The exact same script is used for the leaderboard evaluation computation.

### How to use
Use the `get_froc_vals()`

1. Put the ground truth json files in the folder `evaluation/ground_truth/` with the file name format `case-id_inflammatory-cells.json`,
`case-id_lympocytes.json` and `case-id_monocytes.json` for the respective cell types. These files are provided along with the
`xml` files for the ground truth annotations ([how to access the data](https://monkey.grand-challenge.org/dataset-details/)).

2. Put the output of your algorithm in the folder `evaluation/test/` in a separate folder for each case with a subfolder `output` i.e. `case-id/output/` 
as folder names. In each of these folders, put the json files with the detection output and the name format 
`detected-inflammatory-cells.json`, `detected-lympocytes.json` and `detected-monocytes.json` for the respective cell types.
Additionally, you will need to provide the json file `evaluation/test/output/predictions.json`, which helps to distribute
the jobs.

3. Run `evaluation.py`. The script will compute the evaluation metrics for each case as well as overall and save them to 
`evaluation/test/output/metrics.json`.

The examples provided are the three files that are used for the evaluation phase of the debugging phase.

```angular2html
.
├── ground_truth/
│   ├── A_P000001_inflammatory-cells.json
│   ├── A_P000001_lymphocytes.json
│   ├── A_P000001_monocytes.json
│   └── (...)
└── test/
    ├── input/
    │   ├── A_P000001/
    │   │   └── output/
    │   │       ├── detected-inflammatory-cells.json
    │   │       ├── detected-lymphocytes.json
    │   │       └── detected-monocytes.json
    │   ├── A_P000002/
    │   │   └── (...)
    │   └── (...)
    ├── predictions.json
    └── output/
        └── metrics.json
```