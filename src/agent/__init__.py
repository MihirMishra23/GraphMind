"""Agent package housing various agent strategies."""

from __future__ import annotations

import argparse
from typing import Optional, List

import torch

from llm import LLM, LlamaLLM, OpenAILLM
from .base import BaseAgent
from .llm_agent import LLMAgent
from .kg_llm_agent import KGLLMAgent
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


# TODO: for the walkthroughagent add an additional argument called walkthrough that's a model parameter and is passed in as input to walkthroughagent
def build_agent(
    name: str,
    args: argparse.Namespace,
    llm_client: Optional[LLM],
    walkthrough: Optional[List] = None,
) -> BaseAgent:
    llm_backend = getattr(args, "llm_backend", "llama")

    if name == "walkthrough":
        assert walkthrough
        return WalkthroughAgent(walkthrough)
    if name == "llm":
        if llm_client is None:
            if llm_backend == "openai":
                llm_client = OpenAILLM(
                    model=getattr(args, "openai_model", "gpt-4o-mini"),
                    api_key=getattr(args, "openai_api_key", None),
                )
            else:
                device_map = resolve_device_map(args.device_map)
                llm_client = LlamaLLM(
                    model_id=args.model_id,
                    device_map=device_map,
                    dtype=args.dtype,
                )
        return LLMAgent(
            llm=llm_client,
            memory_mode="none" if args.disable_memory_mode else "llama",
            extraction_max_tokens=args.extract_max_tokens,
            extraction_mode=args.extraction_mode,
        )
    if name in {"kg-llm", "llm-kg"}:
        if llm_client is None:
            if llm_backend == "openai":
                llm_client = OpenAILLM(
                    model=getattr(args, "openai_model", "gpt-4o-mini"),
                    api_key=getattr(args, "openai_api_key", None),
                )
            else:
                device_map = resolve_device_map(args.device_map)
                llm_client = LlamaLLM(
                    model_id=args.model_id,
                    device_map=device_map,
                    dtype=args.dtype,
                )
        return KGLLMAgent(
            llm=llm_client,
            memory_mode="none" if args.disable_memory_mode else "llama",
            extraction_max_tokens=args.extract_max_tokens,
            extraction_mode=args.extraction_mode,
        )

    raise ValueError(f"Unsupported agent type: {name}")


__all__ = [
    "BaseAgent",
    "WalkthroughAgent",
    "LLMAgent",
    "KGLLMAgent",
    "build_agent",
    "resolve_device_map",
]
