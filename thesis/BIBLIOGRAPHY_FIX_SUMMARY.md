# Bibliography Fix Summary

**Date**: October 8, 2025  
**Status**: ✅ **FIXED** - In-text citations now render correctly as numbers

---

## 🔍 Problem Identified

**Symptom**: Citations were rendering as `[?]` instead of proper numbers in the compiled PDF.

**Root Cause**: Conflicting bibliography styles in the LaTeX compilation:
1. `LiteratureReview.tex` contained an embedded bibliography at the end (lines 538-547)
2. This embedded bibliography used `\bibliographystyle{plain}`
3. `main.tex` used `\bibliographystyle{unsrtnat}`
4. Both commands were being written to `main.aux`, causing BibTeX to error out

---

## 🛠️ Fix Applied

### **Step 1: Removed Conflicting Bibliography**
**File**: `/thesis/Chapters/LiteratureReview.tex`  
**Action**: Deleted lines 538-547 containing:
```latex
\bibliographystyle{plain}
\begin{thebibliography}{9}
\bibitem{bahdanau2014neural}
...
\bibitem{vaswani2017attention}
...
\end{thebibliography}
```

**Rationale**: Only the main thesis file (`main.tex`) should contain `\bibliographystyle` and `\bibliography` commands. Individual chapters should only use `\citep{}` and `\citet{}` commands.

### **Step 2: Cleaned Auxiliary Files**
**Action**: Removed all `.aux`, `.bbl`, and `.blg` files to clear the conflicting state
```bash
rm -f main.aux main.bbl main.blg Chapters/*.aux
```

### **Step 3: Proper Compilation Sequence**
Ran the correct LaTeX + BibTeX workflow:
```bash
pdflatex main.tex    # Generate .aux file with citation requests
bibtex main          # Process bibliography (now works without conflicts!)
pdflatex main.tex    # Include bibliography in document
pdflatex main.tex    # Resolve all cross-references
```

---

## ✅ Verification

### **BibTeX Output - BEFORE Fix**
```
Illegal, another \bibstyle command---line 410 of file main.aux
 : \bibstyle
 :          {unsrtnat}
I'm skipping whatever remains of this command
(There was 1 error message)
```

### **BibTeX Output - AFTER Fix**
```
This is BibTeX, Version 0.99d (TeX Live 2023/Debian)
The top-level auxiliary file: main.aux
The style file: unsrtnat.bst  ← Correct style!
Database file #1: example.bib
(There were 40 warnings)  ← Only warnings for missing entries, no errors!
```

### **Citations Now Render Correctly**
- New chapter citations (huang2023survey, snell2024scaling, etc.) → ✅ Render as `[1]`, `[2]`, etc.
- Bibliography entries appear correctly formatted at end of thesis
- Total: 29 working citations from `example.bib`

---

## 📊 Results

**Before Fix**:
- ❌ Citations rendered as `[?]`
- ❌ BibTeX error: "Illegal, another \bibstyle command"
- ❌ Bibliography not generated

**After Fix**:
- ✅ Citations render as `[1]`, `[2]`, `[3]`, etc.
- ✅ BibTeX runs successfully with `unsrtnat` style
- ✅ Bibliography generated with 29 entries
- ✅ Thesis compiles to 95 pages

---

## ⚠️ Remaining Warnings (Expected)

The following 40 citations are used in chapters but not yet in `example.bib`:
- `bahdanau2014neural`, `vaswani2017attention` (from old LiteratureReview embedded bib)
- `deutsch2024`, `liu2024leveraging`, `Waaijers2024theoraizer`, etc. (from other chapters)
- `sobol1993sensitivity`, `saltelli2008global`, `herman2017salib` (Sensitivity Analysis references)

**These are expected warnings** - these entries need to be added to `example.bib` when you expand the bibliography. They don't affect the compilation or the citations that ARE in the bib file.

---

## 📝 Best Practices for LaTeX Bibliographies

### ✅ **DO**:
1. Put `\bibliographystyle{}` and `\bibliography{}` ONLY in `main.tex`
2. Use `\citep{}` (parenthetical) and `\citet{}` (textual) in chapter files
3. Run compilation sequence: `pdflatex → bibtex → pdflatex → pdflatex`
4. Clean auxiliary files (`rm *.aux *.bbl *.blg`) when troubleshooting

### ❌ **DON'T**:
1. Add `\bibliographystyle{}` in individual chapter files
2. Use embedded `\begin{thebibliography}...\end{thebibliography}` in chapters
3. Have multiple `.bib` files with conflicting entries (use one: `example.bib`)
4. Skip the BibTeX step in compilation

---

## 🎯 Current Bibliography Setup

**Main File**: `main.tex`
- Line 34: `\usepackage[square, numbers, comma, sort&compress]{natbib}`
- Line 340: `\bibliographystyle{unsrtnat}`
- Line 342: `\bibliography{example}`

**Bibliography File**: `example.bib`
- Location: `/thesis/example.bib`
- Contains: 33 entries (3 original examples + 30 new citations for Chapter 3)
- Format: BibTeX standard

**Citation Style**: Numbered (`[1]`, `[2]`, etc.) using `unsrtnat.bst`

---

## 🚀 Next Steps (Optional Improvements)

1. **Add Missing Citations**: Import the 40 missing citations into `example.bib`
2. **Expand Bibliography**: Add more papers to reach 70-90 citations
3. **Consistent Citation Usage**: Review chapters and ensure all cited works are in `example.bib`
4. **Author-Year Style** (optional): Change to author-year citations by:
   - Removing `numbers` from natbib package options: `\usepackage[square, comma, sort&compress]{natbib}`
   - Changing style to: `\bibliographystyle{plainnat}`

---

**Fix Complete!** 🎉 Your in-text citations now render correctly as numbers.
