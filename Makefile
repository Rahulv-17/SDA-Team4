.PHONY: start demo up down run-broker run-producer run-consumer run-dashboard test experiments report clean

# One-Command Launcher (Whole Pipeline)
start: demo
demo:
	python run_demo.py

# Phase 0 & Infrastructure commands
up:
	@docker compose up -d || python src/broker/run_broker.py --daemon

down:
	@docker compose down || python -c "print('Stopping local services')"

run-broker:
	python src/broker/run_broker.py

# Phase 1 & Running Streaming Pipeline
run-producer:
	python src/producer/producer.py --stream SEA_recurring --rate 100 --seed 42

run-consumer:
	python src/consumer/consumer.py --stream SEA_recurring --method E

run-dashboard:
	python src/dashboard/server.py --port 8501

# Testing
test:
	pytest tests/ -v

# Experiments & Analysis
experiments:
	python experiments/run_all.py

report:
	python analysis/make_report_assets.py

clean:
	rm -rf __pycache__ */__pycache__ results/*.csv results/*.db
