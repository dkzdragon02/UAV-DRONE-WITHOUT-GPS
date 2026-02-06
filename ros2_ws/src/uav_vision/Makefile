# Makefile for uav_vision package

.PHONY: help test lint format clean install

help:
	@echo "Available targets:"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linters"
	@echo "  format   - Format code with black and isort"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install package"

test:
	pytest test/ -v --cov=uav_vision --cov-report=term-missing

test-html:
	pytest test/ -v --cov=uav_vision --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 uav_vision/ --max-line-length=100
	pylint uav_vision/ --rcfile=.pylintrc || true
	mypy uav_vision/ --ignore-missing-imports || true

format:
	black uav_vision/ test/ --line-length=100
	isort uav_vision/ test/ --profile=black --line-length=100

format-check:
	black uav_vision/ test/ --check --line-length=100
	isort uav_vision/ test/ --check-only --profile=black

clean:
	rm -rf build/ install/ log/ .pytest_cache/ htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

install:
	cd ../.. && source /opt/ros/humble/setup.bash && colcon build --packages-select uav_vision

