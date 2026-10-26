# Cross-Generator Evaluation of AI-Generated Music Detection

ICT-4442 Deep Learning Mini Project, MIT Manipal.

Measuring how much of the reported accuracy in AI-generated music
detection survives honest evaluation: a leave-one-generator-out
protocol and a format-level shortcut audit across four architecture
families.

## Team and ownership

| Member | Reg. No. | Model | Experiment / other |
|---|---|---|---|
| Akshat Yuvan | 230911192 | M4 (AST) | E1 leave-one-generator-out; attention rollout; results consolidation |
| Apratim Sharma | 230911390 | M3 (CRNN) | E2 shortcut audit; manifest and splits |
| Dhanunjay Rahul Varma | 220911628 | M2 (CNN) | Grad-CAM interpretability |
| Ravi Sharma | 230953324 | M1 (MLP) | Literature review (E3 attribution dropped, see below) |

All work is committed from a single account with faculty approval.
Authorship of each notebook is recorded in its commit message.

## Notebooks

`notebooks/` holds the interactive Kaggle notebooks from the interim
phase. `kaggle/` holds the final-phase notebooks, each pushed and run as
a Kaggle kernel; `output/` beside each is exactly what the run returned.

| Notebook | Author | Purpose |
|---|---|---|
| `notebooks/01-setup-and-data-acquisition` | Akshat Yuvan | Zenodo download, archive verification, dataset persistence |
| `notebooks/02-manifest-and-splits` | Apratim Sharma | Authentic-class recovery, format normalisation, manifest, identifier-level splits |
| `notebooks/03-m1-mlp` | Ravi Sharma | M1 feature extraction and training |
| `notebooks/04-m2-cnn` | Dhanunjay Rahul Varma | M2 log-mel precomputation and training |
| `kaggle/03b-m1-mlp-rerun` | Ravi Sharma | M1 rerun so the results file is written by the script |
| `kaggle/05-m3-crnn` | Apratim Sharma | M3 training, test metrics, time-order ablation |
| `kaggle/09-m4-ast` | Akshat Yuvan | M4 fine-tuning, test metrics, attention rollout |
| `kaggle/06-e1-logo` | Akshat Yuvan | E1 for M1–M3 |
| `kaggle/06b-e1-m4` | Akshat Yuvan | E1 for M4 |
| `kaggle/07-e2-shortcut-audit` | Apratim Sharma | E2 for M1–M3 |
| `kaggle/07b-e2-m4` | Apratim Sharma | E2 for M4 |
| `kaggle/08-m2-gradcam` | Dhanunjay Rahul Varma | Grad-CAM on M2, long-term spectrum check |
| `results/consolidate.py` | Akshat Yuvan | Builds every table and summary figure from the saved results files |

## Data

Raw audio is not in this repository. It lives in private Kaggle
datasets: `fakemusiccaps-raw`, `musiccaps-real-16k`, `fmc-manifest`,
`fmc-m1-features`, `fmc-m2-mels`. Results and checkpoints are archived
in `fmc-m3-results`, `fmc-m4-results`, `fmc-e1-results`,
`fmc-e2-results` and `fmc-gradcam-results`.

## Results

Balanced subsample of 10,697 clips, splits by YouTube ID (70/15/15),
seed 42, one shared metric implementation. Every number below is read
from a results file by `results/consolidate.py`; the full tables are in
`results/tables/`.

**Closed set** (test split, 1,604 clips):

| Model | Balanced acc. | AUC | EER |
|---|---|---|---|
| M1 (MLP) | 0.9003 | 0.9705 | 0.0970 |
| M2 (CNN) | 0.9732 | 0.9961 | 0.0249 |
| M3 (CRNN) | 0.9869 | 0.9992 | 0.0137 |
| M4 (AST) | 0.9988 | 1.0000 | 0.0012 |

**E1, leave one generator out** (mean over MusicGen_medium,
stable_audio_open and mustango withheld): open-set balanced accuracy
M1 0.692, M2 0.711, M3 0.676, M4 0.792.

**E2, test-time shortcut audit:** a 7 kHz low-pass drops M1 from 0.895
to 0.504 (chance) and M2–M4 by 5–8 points; MP3 at 128 kbps costs M4 1.6
points but makes M3 call half the real clips fake.

**Scope.** Three of the five E1 folds were run and E3 (six-class
attribution) was dropped, both as declared in the interim report.
