PYTHON ?= .venv/bin/python

.PHONY: doctor test verify-public agent-analysis tables figures release-check

doctor:
	"$(PYTHON)" -c "import numpy, pandas, scipy, sklearn, yaml; print('OpenCoderX offline environment: OK')"

test:
	PYTHONPATH=. "$(PYTHON)" -m pytest -q

verify-public:
	PYTHONPATH=. "$(PYTHON)" scripts/verify_public_artifact.py

agent-analysis:
	MPLCONFIGDIR=/tmp/opencoderx-mpl PYTHONPATH=. "$(PYTHON)" human_study/gateway_agent_v1/analyze_results.py

tables:
	PYTHONPATH=. "$(PYTHON)" scripts/build_tosem_publication_tables.py

figures:
	MPLCONFIGDIR=/tmp/opencoderx-mpl PYTHONPATH=. "$(PYTHON)" scripts/plot_tosem_confirmatory_figures.py

release-check: doctor test verify-public agent-analysis tables figures verify-public
