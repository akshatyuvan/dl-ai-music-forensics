# Cross-Generator Evaluation of AI-Generated Music Detection

ICT-4442 Deep Learning Mini Project, MIT Manipal.

Measuring how much of the reported accuracy in AI-generated music
detection survives honest evaluation: a leave-one-generator-out
protocol and a format-level shortcut audit across four architecture
families.

## Team and ownership

| Member | Reg. No. | Model | Experiment |
|---|---|---|---|
| Akshat Yuvan | 230911192 | M4 (AST) | E1 leave-one-generator-out |
| Apratim Sharma | 230911390 | M3 (CRNN) | E2 shortcut audit |
| Dhanunjay Rahul Varma | 220911628 | M2 (CNN) | Grad-CAM interpretability |
| Ravi Sharma | 230953324 | M1 (MLP) | E3 attribution |

All work is committed from a single account with faculty approval.
Authorship of each notebook is recorded in the commit message and in
`docs/contributions.md`.

## Notebooks

| Notebook | Author | Purpose |
|---|---|---|
| `01-setup-and-data-acquisition` | Akshat Yuvan | Zenodo download, archive verification, dataset persistence |
| `02-manifest-and-splits` | Apratim Sharma | Authentic-class recovery, format normalisation, manifest, identifier-level splits |
| `03-m1-mlp` | Ravi Sharma | M1 feature extraction and training |
| `04-m2-cnn` | Dhanunjay Rahul Varma | M2 log-mel precomputation and training |

## Data

Raw audio is not in this repository. It lives as four private Kaggle
datasets: `fakemusiccaps-raw`, `musiccaps-real-16k`, `fmc-manifest`,
`fmc-m1-features`, `fmc-m2-mels`.

## Results so far

| Model | Balanced acc. | AUC | EER |
|---|---|---|---|
| M1 (MLP) | 0.8972 | 0.9692 | 0.0983 |
| M2 (CNN) | 0.9732 | 0.9961 | 0.0249 |

Closed-set protocol, balanced subsample of 10,697 clips, seed 42.