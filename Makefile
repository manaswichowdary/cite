.PHONY: test check ask serve planted questions holdout audit readme rag-setup recheck

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

rag-setup:
	pip install -r requirements-rag.txt

# Check the same saved answers again with each way of finding evidence (only the checker runs)
recheck:
	python -m cite.evals recheck --results evals/results_holdout/questions.jsonl --retriever words
	python -m cite.evals recheck --results evals/results_holdout/questions.jsonl --retriever embeddings
	python -m cite.evals recheck --results evals/results_holdout/questions.jsonl --retriever hybrid
