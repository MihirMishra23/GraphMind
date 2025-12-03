"""Agent package housing various agent strategies."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from llm import LLM, LlamaLLM
from .base import BaseAgent
from .llm_agent import LLMAgent
from .walkthrough_agent import WalkthroughAgent


def resolve_device_map(requested: str) -> str:
    """Pick device map with priority: cuda > mps > cpu, unless explicitly set."""
    req = requested.lower()
    if req != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_agent(
    name: str, args: argparse.Namespace, llm_client: Optional[LLM]
) -> BaseAgent:
    if name == "walkthrough":
        return WalkthroughAgent()
    if name == "llm":
        if llm_client is None:
            device_map = resolve_device_map(args.device_map)
            llm_client = LlamaLLM(
                model_id=args.model_id,
                device_map=device_map,
                dtype=args.dtype,
            )
        return LLMAgent(
            llm=llm_client,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature,
            memory_mode=args.memory_mode,
            extraction_max_tokens=args.extract_max_tokens,
            extraction_temperature=args.extract_temperature,
            extraction_mode=args.extraction_mode,
        )

    raise ValueError(f"Unsupported agent type: {name}")


__all__ = [
    "BaseAgent",
    "WalkthroughAgent",
    "LLMAgent",
    "build_agent",
    "resolve_device_map",
]
