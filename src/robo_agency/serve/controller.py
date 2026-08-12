"""Двухконтурный рантайм на vLLM.

Контур решений — база с включённым LoRA-адаптером.
Речевой контур — та же база с ВЫКЛЮЧЕННЫМ адаптером.

Решение принимается в два шага:
  1. Ограниченный выбор из {ACT, WAIT, OBSERVE} с логвероятностями. Отсюда
     берётся настоящая калиброванная уверенность и сравнивается с порогом.
  2. Только если решение ACT прошло порог — полная генерация по JSON-схеме.

Второй шаг дороже первого, поэтому на WAIT-ситуациях (а их около половины)
он вообще не выполняется. Это же даёт честную вероятность для калибровки:
поле `confidence` внутри JSON обучено на константах и калиброванным не является.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from ..prompts import build_messages
from ..schema import DecisionType, RobotDecision, Situation, decision_json_schema

logger = logging.getLogger(__name__)

DECISION_CHOICES = [d.value for d in DecisionType]


@dataclass(slots=True)
class ControllerConfig:
    base_model: str = "Qwen/Qwen3-8B"
    adapter_path: str | None = "outputs/adapter-decisions"
    act_threshold: float = 0.5
    temperature_calibration: float = 1.0
    max_decision_tokens: int = 256
    max_speech_tokens: int = 256
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096


@dataclass(slots=True)
class DecisionOutcome:
    decision: RobotDecision
    act_probability: float
    crossed_threshold: bool


class RoboController:
    def __init__(self, config: ControllerConfig) -> None:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        self.llm = LLM(
            model=config.base_model,
            enable_lora=config.adapter_path is not None,
            max_lora_rank=64,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
        )
        self._lora = (
            LoRARequest("decisions", 1, config.adapter_path) if config.adapter_path else None
        )
        self._schema = decision_json_schema()

    # --- внутреннее ---------------------------------------------------------

    def _render(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompt: str, sampling_params, use_adapter: bool):
        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=self._lora if use_adapter else None,
        )
        return outputs[0].outputs[0]

    def _act_probability(self, prompt: str) -> tuple[str, float]:
        """Шаг 1: выбор из трёх решений с логвероятностями."""
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
            logprobs=len(DECISION_CHOICES) + 4,
            guided_decoding=GuidedDecodingParams(choice=DECISION_CHOICES),
        )
        output = self._generate(prompt, params, use_adapter=True)
        chosen = output.text.strip()

        probability = self._extract_act_probability(output, chosen)
        return chosen, probability

    def _extract_act_probability(self, output: Any, chosen: str) -> float:
        """Достаёт P(ACT) из логвероятностей первого токена.

        Если логвероятности недоступны, возвращается вырожденная оценка по
        выбранному варианту — это честнее, чем выдумать число.
        """
        logprobs = getattr(output, "logprobs", None)
        if not logprobs:
            return 1.0 if chosen == DecisionType.ACT.value else 0.0

        first_position = logprobs[0]
        act_logprob = None
        total = 0.0
        for entry in first_position.values():
            token = getattr(entry, "decoded_token", None) or ""
            probability = math.exp(entry.logprob)
            total += probability
            if token.strip().upper().startswith("ACT"):
                act_logprob = probability

        if act_logprob is None:
            return 1.0 if chosen == DecisionType.ACT.value else 0.0
        return act_logprob / total if total else act_logprob

    # --- публичный интерфейс ------------------------------------------------

    def decide(self, situation: Situation) -> DecisionOutcome:
        """Контур решений: что робот делает прямо сейчас."""
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        from ..evaluation.calibration import apply_temperature

        prompt = self._render(build_messages(situation))
        chosen, raw_probability = self._act_probability(prompt)
        probability = apply_temperature(raw_probability, self.config.temperature_calibration)

        crossed = probability >= self.config.act_threshold
        if chosen != DecisionType.ACT.value or not crossed:
            decision_type = (
                DecisionType(chosen) if chosen in DECISION_CHOICES else DecisionType.WAIT
            )
            if decision_type is DecisionType.ACT:
                decision_type = DecisionType.WAIT  # решение не прошло порог
            return DecisionOutcome(
                decision=RobotDecision(decision=decision_type, confidence=probability),
                act_probability=probability,
                crossed_threshold=crossed,
            )

        # Шаг 2: полное решение под JSON-схемой.
        params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_decision_tokens,
            guided_decoding=GuidedDecodingParams(json=self._schema),
        )
        output = self._generate(prompt, params, use_adapter=True)
        payload = json.loads(output.text)
        payload["confidence"] = probability  # заменяем необученное поле калиброванным
        decision = RobotDecision.model_validate(payload)
        return DecisionOutcome(decision=decision, act_probability=probability, crossed_threshold=True)

    def speak(self, messages: list[dict[str, str]]) -> str:
        """Речевой контур: адаптер выключен, работает чистая база."""
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=self.config.max_speech_tokens)
        output = self._generate(self._render(messages), params, use_adapter=False)
        return output.text.strip()
