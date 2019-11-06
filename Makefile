help:
	@echo 'Makefile for ts                        '
	@echo '                                                                   '
	@echo 'To use, enter one of the following commands in your terminal:      '
	@echo ' make test             run tests using pytest                      '
	@echo ' make test-cov         run tests with coverage                     '
	@echo ' make test-covhtml     run tets with coverage and view results     '
	@echo '                                                                   '

test:
	pytest

test-cov:
	pytest --cov ts

test-covhtml:
	pytest --cov ts --cov-report html && open ./htmlcov/index.html
