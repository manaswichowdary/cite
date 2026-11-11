.PHONY: test check ask serve planted questions holdout audit readme

test:
	python -m pytest -q

check:
	python -m cite.check

ask:
	python -m cite.ask "$(Q)"

serve:
	python -m cite.api

planted:
	python -m cite.evals planted

questions:
	python -m cite.evals questions $(ARGS)

holdout:
	python -m cite.evals questions --questions evals/holdout_questions.json --out evals/results_holdout --max-usd 0.50

audit:
	python -m cite.evals audit --limit 10

readme:
	python -m cite.evals readme
