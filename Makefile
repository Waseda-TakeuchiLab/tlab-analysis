test: mypy unittest

mypy:
	mypy .

unittest:
	coverage run -m unittest
	coverage html
	coverage report
