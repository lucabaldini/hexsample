all:

test:
	pytest tests -s

html:
	cd docs; make html;

clean:
	rm -rf hexsample/__pycache__ tests/__pycache__ docs/_build
