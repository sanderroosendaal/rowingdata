# Coverage benchmark

Machine-generated **line coverage** and a concise **feature** (capability) map for the `rowingdata` package. Re-run after major changes.

## How to reproduce

```bash
pip install pytest-cov
cd /path/to/rowingdata
python -m pytest tests/ --cov=rowingdata --cov-report=term-missing:skip-covered --cov-report=json:coverage.json
```

Optional HTML report: add `--cov-report=html:htmlcov` (see `.gitignore` for `htmlcov/`).

## Last run (committed benchmark)

| Metric | Value |
|--------|-------|
| **Date** | 2026-03-26 |
| **Python** | 3.9 (Windows; local run) |
| **Tests** | `tests/test_rowingdata.py`, `tests/testparser.py` — **144 passed** |
| **Statements (in `rowingdata/`)** | 6770 |
| **Lines covered** | 5118 |
| **Line coverage** | **~76%** (`75.6%` exact) |

Coverage measures executable lines under `rowingdata/` when the test suite imports and exercises code. It does **not** include `tests/` in the numerator (pytest-cov uses `--source=rowingdata`).

## Line coverage by module

Sorted by relevance; **0%** = no statements executed during tests (typically CLI plot scripts that only run as `__main__`).

| Module | Stmts | Miss | Cover |
|--------|------:|-----:|------:|
| `rowingdata.py` | 2350 | 112 | **95%** |
| `csvparsers.py` | 1339 | 42 | **97%** |
| `otherparsers.py` | 239 | 8 | **97%** |
| `fitwrite.py` | 668 | 103 | **85%** |
| `fitwrite_spec.py` | 123 | 27 | **78%** |
| `utils.py` | 89 | 19 | **79%** |
| `checkdatafiles.py` | 134 | 31 | **77%** |
| `writetcx.py` | 236 | 65 | **72%** |
| `trainingparser.py` | 232 | 73 | **69%** |
| `tcxtools.py` | 312 | 162 | **48%** |
| `gpxtools.py` | 49 | 32 | **35%** |
| `gpxwrite.py` | 100 | 79 | **21%** |
| `__main__.py`, `boatedit.py`, `copystats.py`, `crewnerdplot*.py`, `ergdata*.py`, `ergstick*.py`, `konkatenaadje.py`, `laptesting.py`, `obsolete.py`, `painsled*.py`, `roweredit.py`, `rowpro*.py`, `speedcoach*.py`, `tcxplot*.py`, `tcxtoc2.py`, `windcorrected.py` | various | all | **0%** |

**Total** package line coverage: **76%** (see `coverage.json` for per-line detail).

## Feature coverage (capability vs tests)

“Feature” here means **user-visible capability**; **coverage** is the measured line % for modules that implement it.

| Capability | Automated tests (examples) | Module line coverage |
|------------|----------------------------|----------------------|
| **Core session object** (`rowingdata`, intervals, stats, plots mocked) | `TestBasicRowingData`, `test_intervals_rowingdata`, `test_plot_*` | `rowingdata.py` **95%** |
| **CSV / file format parsers** | `test_check` (many formats), parser tests in `testparser.py` | `csvparsers.py` **97%**, `otherparsers.py` **97%** |
| **FIT read / write** | `TestFITParser`, `exporttofit`, `FITParser`, instroke tests | `fitwrite.py` **85%**, `fitwrite_spec.py` **78%** |
| **TCX / GPX I/O** | `test_read_tcx`, `test_write_tcx`, `test_write_csv` | `writetcx.py` **72%**, `tcxtools.py` **48%**, `gpxwrite.py` **21%**, `gpxtools.py` **35%** |
| **Training string parser** | `TestStringParser::teststringparser` | `trainingparser.py` **69%** |
| **Utilities** | indirect via rowingdata | `utils.py` **79%** |
| **checkdatafiles** | `test_check` | **77%** |
| **CLI plot scripts** (CrewNerd, PainSled, TCX, erg, etc.) | *none* — entry points not invoked as subprocess | **0%** on each `*plot*.py`, `*toc2*.py`, etc. |
| **Obsolete / laptesting** | *none* | `obsolete.py`, `laptesting.py` **0%** |

### Gaps (high level)

1. **Plot / conversion CLIs**: Almost all standalone scripts are uncovered; line coverage is **0%** unless imported elsewhere.
2. **GPX write path**: Lower than TCX (`gpxwrite.py` **21%**).
3. **`tcxtools`**: Large surface (**48%**); many branches unused in tests.
4. **`fitwrite` / `fitwrite_spec`**: Remaining misses are mostly branches (GPS, multi-lap edge cases, companion errors).

## Historical CI

`.travis.yml` runs `nosetests` on `tests/testparser.py` and `tests/test_rowingdata.py` and may upload to Coveralls using `.coveragerc`. This document’s numbers are from **pytest + pytest-cov** locally and may differ slightly from CI.
