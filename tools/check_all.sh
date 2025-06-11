#!/bin/bash
echo "✅ Linting mit flake8"
flake8 .

echo "✅ Formatieren mit black"
black .

echo "✅ Sortieren mit isort"
isort .

echo "✅ Typprüfung mit mypy"
mypy .

echo "✅ Tests mit pytest"
pytest --cov=detectors
