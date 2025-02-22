thesis:
	xelatex GLUTthesis
	biber GLUTthesis
	xelatex GLUTthesis

# Only use in Unix
clean:
	find . -name '*.aux' -print0 | xargs -0 rm -rf
	rm -rf *.lof *.log *.lot *.out *.toc *.bbl *.blg *.thm *.nav *.xml *.snm *.bcf
