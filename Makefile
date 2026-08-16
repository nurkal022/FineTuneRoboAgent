# Удобный запуск пайплайна. Профиль по умолчанию — RTX 5080 (16 ГБ).
# Другой профиль: make sft SFT_CONFIG=configs/sft.yaml

PY ?= python
DATA_CONFIG ?= configs/data_mix.yaml
SFT_CONFIG ?= configs/sft_rtx5080.yaml
DPO_CONFIG ?= configs/dpo_rtx5080.yaml
DATA_DIR ?= data/processed
BASE_MODEL ?= Qwen/Qwen3-8B
ASR_CONFIG ?= configs/whisper_kk.yaml
LOG_DIR ?= logs

.DEFAULT_GOAL := help
.PHONY: help setup doctor preflight test fetch data sft dpo pairs retention inspect clean all \
	asr-peek asr-baseline asr-train asr-eval

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Первый запуск: setup -> preflight -> fetch -> data -> sft"

setup:  ## Установить окружение на сервере (Ubuntu + Blackwell)
	bash scripts/setup_server.sh

doctor:  ## Починить torch, если установка подменила его CPU-сборкой
	@$(PY) scripts/verify_torch.py || ( \
		echo "Переустанавливаю torch из CUDA-индекса..."; \
		uv pip install --reinstall torch torchvision --torch-backend=auto && \
		$(PY) scripts/verify_torch.py )

preflight:  ## Проверить GPU, стек и оценить VRAM по конфигу
	$(PY) scripts/preflight.py --config $(SFT_CONFIG)

test:  ## Прогнать тесты
	$(PY) -m pytest tests -q

inspect:  ## Показать реальные колонки датасета: make inspect DS=thunlp/ProactiveAgent
	@test -n "$(DS)" || (echo "Укажите DS=<путь к датасету>"; exit 1)
	$(PY) -m robo_agency.cli inspect-dataset $(DS)

fetch:  ## Скачать корпус проактивности (ProactiveBench)
	$(PY) scripts/fetch_data.py

data: 	## Собрать обучающий корпус из готовых датасетов
	$(PY) -m robo_agency.cli build-data --config $(DATA_CONFIG) --output $(DATA_DIR)

sft: preflight  ## Этап 2: обучить адаптер решений
	@mkdir -p $(LOG_DIR)
	PYTHONUNBUFFERED=1 $(PY) -m robo_agency.cli train-sft --config $(SFT_CONFIG) 2>&1 | tee $(LOG_DIR)/sft.log

pairs:  ## Этап 4: собрать пары предпочтений: make pairs IN=logs/interactions.jsonl
	@test -n "$(IN)" || (echo "Укажите IN=<лог взаимодействий>"; exit 1)
	$(PY) -m robo_agency.cli build-pairs --input $(IN) \
		--output $(DATA_DIR)/preference_pairs.jsonl

dpo:  ## Этап 5: DPO на неявной обратной связи
	@mkdir -p $(LOG_DIR)
	PYTHONUNBUFFERED=1 $(PY) -m robo_agency.cli train-dpo --config $(DPO_CONFIG) 2>&1 | tee $(LOG_DIR)/dpo.log

retention:  ## Этапы 0 и 6: замер сохранения речевых способностей
	$(PY) -m robo_agency.cli retention --base $(BASE_MODEL) \
		--adapter outputs/adapter-decisions \
		--eval-file $(DATA_DIR)/retention_eval.jsonl

asr-peek:  ## Речь: осмотреть казахский корпус, не скачивая целиком
	$(PY) scripts/asr.py --config $(ASR_CONFIG) peek

asr-baseline:  ## Речь: замерить Whisper БЕЗ дообучения (точка отсчёта)
	@mkdir -p $(LOG_DIR) docs/experiments/asr
	$(PY) scripts/asr.py --config $(ASR_CONFIG) baseline \
		--json docs/experiments/asr/baseline.json 2>&1 | tee $(LOG_DIR)/asr_baseline.log

asr-train:  ## Речь: дообучить Whisper на казахском
	@mkdir -p $(LOG_DIR)
	PYTHONUNBUFFERED=1 $(PY) scripts/asr.py --config $(ASR_CONFIG) train 2>&1 | tee $(LOG_DIR)/asr_train.log

asr-eval:  ## Речь: замерить дообученную модель
	@mkdir -p $(LOG_DIR) docs/experiments/asr
	$(PY) scripts/asr.py --config $(ASR_CONFIG) eval \
		--json docs/experiments/asr/after.json 2>&1 | tee $(LOG_DIR)/asr_eval.log

all: data sft  ## Собрать данные и обучить адаптер

clean:  ## Удалить кеши сборки и тестов
	rm -rf .pytest_cache **/__pycache__ *.egg-info
