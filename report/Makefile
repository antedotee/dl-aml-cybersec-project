# Build PDF from LaTeX (run from this directory: cd report && make)
PDF = main.pdf
TEX = main.tex
BIB = references.bib

.PHONY: all clean figures distclean

all: figures $(PDF)

figures:
	@python3 scripts/plot_ablation_figure.py

$(PDF): $(TEX) $(BIB) figures/generated/ablation_roc_auc.pdf
	pdflatex -interaction=nonstopmode -file-line-error $(TEX)
	bibtex main
	pdflatex -interaction=nonstopmode -file-line-error $(TEX)
	pdflatex -interaction=nonstopmode -file-line-error $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot

distclean: clean
	rm -f $(PDF)
