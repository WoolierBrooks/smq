.PHONY: install test figures synthetic benchmark quickstart assets

help:
	@echo "make install | test | figures | synthetic | benchmark | quickstart | assets"

install:
	pip install -e ".[experiments,analysis,dev]"

test:
	pytest -v

figures:
	python analysis/make_figures.py

synthetic:
	python experiments/generate_synthetic.py --out data/synthetic

benchmark:
	python experiments/run_benchmark.py --synthetic-dir data/synthetic --local-dir data/multiclass --output results/raw/benchmark.csv

quickstart:
	python examples/quickstart.py

assets:
	python docs/make_assets.py
