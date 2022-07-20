test: flake8 mypy unittest

flake8:
	flake8 .

mypy:
	mypy .

unittest:
	coverage run -m unittest
	coverage html
	coverage report
