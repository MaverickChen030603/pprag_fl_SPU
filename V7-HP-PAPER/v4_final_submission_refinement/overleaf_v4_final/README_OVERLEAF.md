# Overleaf package

Upload every file in this directory to a new Overleaf project and set
`main.tex` as the main document. The package uses only standard TeX Live
packages and BibTeX via `references.bib`.

For a minimal upload, use only `paper_full_v4_final.tex` and
`references.bib`, then set `paper_full_v4_final.tex` as the main document.
The modular `main.tex` version is easier for collaborative editing.

## Venue-template migration

The current `main.tex` is a generic two-column wrapper because the target
conference template has not been fixed. For ACL/EMNLP/NAACL, replace the
document class and preamble with the official template, retain the title and
anonymous author block, then keep these lines in the document body:

```tex
\begin{abstract}
\input{abstract}
\end{abstract}
\input{paper_content}
\bibliography{references}
\appendix
\input{appendix}
```

The figures are editable TikZ/PGFPlots source. Tables use `booktabs`. All
citations resolve to entries in `references.bib`.
