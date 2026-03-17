# Course report (LaTeX → PDF)

**Authors:** Kartik Yadav, Agrapujya Lashkari  
**Institution:** Rishihood University  

**Source:** `main.tex` + `references.bib`  
**Output:** `main.pdf` (generated locally; not committed unless you choose to)

## Install LaTeX (macOS)

**Option A — BasicTeX (smaller, ~100 MB)**

```bash
brew install --cask basictex
# Restart the terminal, then install packages used by this document:
sudo tlmgr update --self
sudo tlmgr install collection-latexextra collection-fontsrecommended lm microtype
```

**Option B — MacTeX (full distribution, multi-GB)**

```bash
brew install --cask mactex-no-gui
```

**Option C — Linux**

```bash
sudo apt-get install texlive-full   # Debian/Ubuntu
# or a smaller metapackage: texlive-latex-extra texlive-fonts-recommended
```

## Build

```bash
cd report
python3 scripts/plot_ablation_figure.py   # bar chart from notebook metrics
make
```

Open `main.pdf`. If `bibtex` or packages are missing, install the package `tlmgr` names the error suggests.

## Figures

| File | Origin |
|------|--------|
| `figures/generated/ablation_roc_auc.pdf` | `scripts/plot_ablation_figure.py` (metrics match executed notebook) |
| `../reports/figures/phase1_nsl_corr.png` | Run `notebooks/phase1_eval_nsl_mae_ocsvm.ipynb` (optional; placeholder box if missing) |

## Git

Per project instructions: **do not commit** until you are ready; add `main.pdf` only if your course allows binary PDFs in the repo.
