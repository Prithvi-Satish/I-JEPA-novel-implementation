# QuadTree-JEPA Preprint Compilation & Source Guide

This directory contains the publication-ready, mathematically rigorous 6–8 page academic preprint for **QuadTree-JEPA (Hierarchical Multi-Scale Self-Supervised Joint-Embedding Predictive Architecture for Fine-Grained Visual Representation)**.

---

## 📁 Directory Structure

```
preprint/
├── main.tex                    # Complete LaTeX preprint document (Two-Column Standard CVPR/IEEE style)
├── references.bib              # Complete BibTeX bibliography with full citations
├── MASTER_PREPRINT_PROMPT.md   # Extensively detailed master prompt for LLMs (to generate/expand sections)
└── README.md                   # This instruction and build guide
```

---

## 🛠️ How to Compile

### Option 1: On Overleaf (Recommended - Zero Setup)
1. Go to [Overleaf](https://www.overleaf.com).
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Upload `main.tex` and `references.bib`.
4. (Optional) Copy image files from `plots/` (`label_efficiency_comparison.png`, `confusion_matrix_finetuned.png`) into the Overleaf project and uncomment the `\includegraphics` lines in `main.tex`.
5. Click **Recompile**.

### Option 2: Local Compilation (Command Line)
Make sure you have `pdflatex` and `bibtex` installed (e.g. via TeX Live or MiKTeX):

```bash
cd preprint
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This will produce `main.pdf`.

---

## 📊 Incorporating Figures from `plots/`
The codebase already generated publication-grade figures in `../plots/`:
- `label_efficiency_comparison.png` $\rightarrow$ Figure 1 (Label efficiency curve)
- `confusion_matrix_finetuned.png` $\rightarrow$ Figure 3 (Confusion matrix across 18 classes)

In `main.tex`, simply replace the placeholder `\fbox{...}` commands with:
```latex
\includegraphics[width=0.95\linewidth]{../plots/label_efficiency_comparison.png}
```
and
```latex
\includegraphics[width=0.95\linewidth]{../plots/confusion_matrix_finetuned.png}
```
