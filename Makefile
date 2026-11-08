PY ?= python

.PHONY: install test lint data reproduce full record quality appendix benchmark
install:
	$(PY) -m pip install -e ../skeptic -e ".[recorder,dev]"
test:
	$(PY) -m pytest
lint:
	ruff check src tests studies
	mypy src
data:                 ## the 3 days used in the README (add more days with --start/--end)
	$(PY) -m execlab.download --start 2026-10-27 --end 2026-10-29 --out ../data/binance
reproduce:            ## headline charts from the checked-in 2-hour sample (< 1 minute)
	$(PY) studies/run_study.py --sample
full:                 ## full study on ../data/binance aggTrades zips (about an hour on 2 cores; trained models are cached)
	$(PY) studies/run_study.py --trials 10
record:               ## live L2 recorder (run for 2-4 weeks on your own machine)
	$(PY) -m execlab.recorder --symbols BTCUSDT ETHUSDT --levels 20 --out data/l2
quality:              ## data quality report for recorded L2 data
	$(PY) -c "from execlab.quality import l2_report, to_markdown; print(to_markdown([l2_report('data/l2', s) for s in ('BTCUSDT','ETHUSDT')]))"
appendix:             ## regenerate docs/appendix.md (the PDF note is checked in)
	$(PY) studies/note_appendix.py
benchmark:
	$(PY) studies/benchmark.py
