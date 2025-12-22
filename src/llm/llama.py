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
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map
        )
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device_map=device_map,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        outputs = self.generator(
            formatted_prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
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
