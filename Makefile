.PHONY: test check ask serve planted questions

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
