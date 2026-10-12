# PowerShell helper script for Windows environments
param (
    [string]$Command = "help"
)

switch ($Command) {
    "start" {
        python run_demo.py
    }
    "demo" {
        python run_demo.py
    }
    "up" {
        Write-Host "Starting message broker and services..." -ForegroundColor Cyan
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            docker compose up -d
        } else {
            Write-Host "Docker not detected. Starting local MQTT broker fallback..." -ForegroundColor Yellow
            Start-Process -FilePath "python" -ArgumentList "src/broker/run_broker.py" -NoNewWindow
        }
    }
    "down" {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            docker compose down
        } else {
            Write-Host "Stopping background Python brokers..." -ForegroundColor Yellow
            Stop-Process -Name "python" -ErrorAction SilentlyContinue
        }
    }
    "run-broker" {
        python src/broker/run_broker.py
    }
    "run-producer" {
        python src/producer/producer.py --stream SEA_recurring --rate 100 --seed 42
    }
    "run-consumer" {
        python src/consumer/consumer.py --stream SEA_recurring --method E
    }
    "run-dashboard" {
        python src/dashboard/server.py --port 8501
    }
    "test" {
        pytest tests/ -v
    }
    "experiments" {
        python experiments/run_all.py
    }
    "report" {
        python analysis/make_report_assets.py
    }
    Default {
        Write-Host "Usage: .\run.ps1 [start|demo|up|down|run-broker|run-producer|run-consumer|run-dashboard|test|experiments|report]"
    }
}
