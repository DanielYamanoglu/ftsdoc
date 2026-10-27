"""LangChain pipeline: ingest -> classify -> structured extraction.

Two interchangeable backends:
  * llm     - LangChain ``with_structured_output`` against Anthropic or OpenAI (needs an API key)
  * offline - fine-tuned NER + regex fills the same Pydantic schema (no keys, no network)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from . import offline
from .classify import classify
from .ingest import load_text
from .ner.infer import NER_DIR, NerModel
from .schemas import SCHEMAS, ExtractionResult

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You extract structured data from a {doc_type} document. Return only values that appear in the "
               "text; use null or an empty list when a field is absent."),
    ("human", "{text}"),
])


def pick_backend(requested: str = "auto") -> str:
    if requested in ("llm", "offline"):
        return requested
    return "llm" if (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")) else "offline"


def make_llm() -> Any:
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("FTSDOC_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("FTSDOC_LLM_MODEL", DEFAULT_OPENAI_MODEL), temperature=0)
    raise RuntimeError("llm backend requires ANTHROPIC_API_KEY or OPENAI_API_KEY")


def llm_extractor(llm: Any) -> Runnable:
    """text + doc_type -> schema instance, via ``llm.with_structured_output``."""
    def run(inp: dict) -> Any:
        chain = PROMPT | llm.with_structured_output(SCHEMAS[inp["doc_type"]])
        return chain.invoke(inp)
    return RunnableLambda(run)


def offline_extractor(ner: Optional[NerModel]) -> Runnable:
    def run(inp: dict) -> Any:
        text = inp["text"]
        ents = offline.regex_entities(text)
        if ner is not None:
            ents = offline.merge(ner.predict(text), ents)
        return offline.fill(inp["doc_type"], text, ents)
    return RunnableLambda(run)


def build_chain(backend: str = "auto", llm: Any = None, ner: Optional[NerModel] = None,
                doc_type: Optional[str] = None) -> Runnable:
    """Build ``path -> ExtractionResult``. ``llm``/``ner`` can be injected (used by tests)."""
    backend = pick_backend(backend)
    if backend == "llm":
        extractor = llm_extractor(llm or make_llm())
    else:
        if ner is None and NerModel.available():
            ner = NerModel(NER_DIR)
        extractor = offline_extractor(ner)

    def load(path: Any) -> dict:
        return {"source": Path(path).name, "text": load_text(path)}

    def route(d: dict) -> dict:
        if doc_type:
            return {**d, "doc_type": doc_type, "confidence": None}
        label, conf = classify(d["text"])
        return {**d, "doc_type": label, "confidence": conf}

    def package(d: dict) -> ExtractionResult:
        return ExtractionResult(source=d["source"], doc_type=d["doc_type"], classifier_confidence=d["confidence"],
                                backend=backend, data=d["data"])

    return (RunnableLambda(load) | RunnableLambda(route)
            | RunnableLambda(lambda d: {**d, "data": extractor.invoke(d)}) | RunnableLambda(package))


def extract(path: str | Path, backend: str = "auto", doc_type: Optional[str] = None) -> ExtractionResult:
    return build_chain(backend, doc_type=doc_type).invoke(path)
