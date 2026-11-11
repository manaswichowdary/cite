# Cite

A research agent that answers questions with sources, then checks that every claim really appears in the source it cites.

Status: early setup. This README grows as each part is built and measured.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add your keys
python -m cite.check     # prints yes or no for each key
python -m pytest -q
```
