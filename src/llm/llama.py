from __future__ import annotations

"""LLaMA client wrapper using Hugging Face transformers (Llama 3.2 3B Instruct)."""
from typing import Iterable, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import LLM


class LlamaLLM(LLM):
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        device_map: str = "auto",
        dtype: str = "auto",
        **kwargs,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map=device_map
            ),
            tokenizer=self.tokenizer,
            device_map=device_map,
            dtype=dtype,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 1.0,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        outputs = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            # temperature=temperature,
            # do_sample=temperature > 0,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"]
        completion = text

        if stop:
            for token in stop:
                idx = completion.find(token)
                if idx != -1:
                    completion = completion[:idx]
                    break
        return completion.strip()
