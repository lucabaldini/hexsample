all:

test:
	pytest tests -s

html:
	cd docs; make html;
