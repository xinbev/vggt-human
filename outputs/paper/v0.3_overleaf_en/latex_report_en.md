# v0.3 English LaTeX Build and Verification Report

## Build

- Date: 2026-09-15
- Main source: `main_en.tex`
- Appendix source: `appendix_en.tex`
- Engine: XeLaTeX + BibTeX
- Output: `main_en.pdf`
- Pages: 28
- Paper size: US Letter (612 x 792 pt)
- Compilation: successful
- Undefined citations/references: 0
- LaTeX/package errors: 0
- Overfull boxes: 0

The source uses TeX Gyre Termes and TeX Gyre Heros when available on Overleaf, with Times New Roman and Arial as local fallbacks. The Overleaf project must use the XeLaTeX compiler.

## Translation Coverage

- Title, abstract, Introduction, Related Work, Method, Experiments, Conclusion, acknowledgments, AI disclosure, Ethics Statement, and Reproducibility Statement are present in English.
- Every appendix section and subsection is translated.
- Figure captions, table captions, table notes, placeholder warnings, and failure-analysis text are translated.
- Equations, numeric values, citation keys, labels, and references are preserved.
- Main-paper labels match the Chinese source.
- Appendix labels match the Chinese source.
- Chinese characters remaining in `main_en.tex` or `appendix_en.tex`: 0.

## Visual Verification

- Page 1: title, real teaser, caption, and abstract render correctly.
- Page 18: the real-world measurement figure wraps on the right without overlap or clipping.
- Appendix Table 3 is readable and keeps the original Bonn values.
- The detailed scale/TRSTR figure is located in the appendix.
- The eight inference steps appear together without interruption by later tables.
- Placeholder tables and the NBAI placeholder figure remain visibly marked and cannot be mistaken for completed experiments.

## Remaining Author Tasks

- Replace every red `To be evaluated` entry with verified server results.
- Replace the placeholder figure with real ablation, robustness, and failure-case panels.
- Verify training hyperparameters, checkpoints, dataset splits, masks, licenses, runtime, and memory.
- Compress the main paper to the final ICLR page limit before submission.
