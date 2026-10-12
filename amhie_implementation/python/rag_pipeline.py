"""
RAG Generation Pipeline for AMHIE.
Integrates AdaptiveRetriever with Qwen2.5-0.5B-Instruct.
Constructs layout-preserving prompts with structured markdown tables and citations.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .retriever import AdaptiveRetriever, RetrievedContext
from .logger import safe_print

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    HAS_TRANSFORMERS = False


@dataclass
class RAGResponse:
    query: str
    answer: str
    contexts: List[RetrievedContext]
    retrieval_paths: List[str]
    latency_ms: float
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": [c.to_dict() for c in self.contexts],
            "retrieval_paths": self.retrieval_paths,
            "latency_ms": round(self.latency_ms, 2),
            "model_used": self.model_used
        }


class AMHIERAGPipeline:
    """
    End-to-End Multimodal Hierarchical RAG Pipeline.
    """
    def __init__(self,
                 retriever: AdaptiveRetriever,
                 model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 load_local_llm: bool = False):
        self.retriever = retriever
        self.model_name = model_name
        self.load_local_llm = load_local_llm
        self._llm_pipeline = None

        if self.load_local_llm:
            self._init_llm()

    def _init_llm(self):
        """Loads Qwen2.5-0.5B-Instruct tokenizer and model weights."""
        if not HAS_TRANSFORMERS:
            safe_print("[AMHIERAGPipeline] Transformers not installed. Using internal extractive generator.")
            return

        try:
            safe_print(f"[AMHIERAGPipeline] Loading generator: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            device = "cuda" if (torch.cuda.is_available()) else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            self._llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )
            safe_print("[AMHIERAGPipeline] Qwen2.5-0.5B-Instruct generator ready.")
        except Exception as e:
            safe_print(f"[AMHIERAGPipeline] Could not load {self.model_name}: {e}. Using fast deterministic generator.")
            self._llm_pipeline = None

    def _build_context_prompt(self, contexts: List[RetrievedContext]) -> str:
        """Formats retrieved multi-granularity nodes into a structured prompt."""
        if not contexts:
            return "No relevant document context found."

        sections = []
        for i, ctx in enumerate(contexts):
            header = f"[Source {i+1} | Page {ctx.page_number} | Type: {ctx.node_type} | ID: {ctx.node_id}]"
            content = ctx.content.strip()
            sections.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(sections)

    def generate(self, query: str) -> RAGResponse:
        """Executes full RAG flow: Adaptive retrieval -> Prompt formatting -> Generation."""
        start_time = time.time()

        # 1. Retrieve adaptive contexts
        contexts = self.retriever.retrieve(query)
        retrieval_paths = [c.traversal_path for c in contexts]

        context_str = self._build_context_prompt(contexts)

        # 2. Generate answer
        if self._llm_pipeline is not None:
            prompt = (
                f"<|im_start|>system\nYou are an expert AI assistant answering questions using "
                f"the provided document hierarchy. Pay close attention to Markdown tables and figures.<|im_end|>\n"
                f"<|im_start|>user\nContext Information:\n{context_str}\n\n"
                f"Question: {query}\n"
                f"Provide a clear, accurate, and direct answer citing relevant pages and tables.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                outputs = self._llm_pipeline(prompt)
                generated_text = outputs[0]["generated_text"]
                answer = generated_text[len(prompt):].strip()
                model_used = self.model_name
            except Exception as e:
                answer = self._extractive_fallback_answer(query, contexts)
                model_used = f"Extractive Fallback ({e})"
        else:
            answer = self._extractive_fallback_answer(query, contexts)
            model_used = "AMHIE Extractive Reasoner (Zero-Dependency)"

        latency = (time.time() - start_time) * 1000.0

        return RAGResponse(
            query=query,
            answer=answer,
            contexts=contexts,
            retrieval_paths=retrieval_paths,
            latency_ms=latency,
            model_used=model_used
        )

    def _extractive_fallback_answer(self, query: str, contexts: List[RetrievedContext]) -> str:
        """
        Extractive synthesis fallback that answers queries using retrieved tables and text.
        """
        if not contexts:
            return "I could not find relevant information in the uploaded document to answer your question."

        lines = []
        lines.append(f"Based on the **AMHIE Adaptive Retrieval** analysis (Page {[c.page_number for c in contexts]}):")
        lines.append("")

        for ctx in contexts:
            if ctx.node_type == "TABLE":
                lines.append(f"**Extracted Table (Page {ctx.page_number} - {ctx.node_id}):**")
                lines.append(ctx.content)
                lines.append("")
            elif ctx.node_type in ["PAGE_COARSE", "TEXT"]:
                # Excerpt key sentences matching query words
                q_words = set(query.lower().split())
                sens = [s.strip() for s in ctx.content.split(".") if s.strip()]
                matched_sens = [s for s in sens if any(w in s.lower() for w in q_words if len(w) > 3)]
                if matched_sens:
                    excerpt = ". ".join(matched_sens[:3]) + "."
                    lines.append(f"- **Page {ctx.page_number} Context:** {excerpt}")
                else:
                    excerpt = ". ".join(sens[:2]) + ("..." if len(sens) > 2 else "")
                    lines.append(f"- **Page {ctx.page_number} Summary:** {excerpt}")

        lines.append("")
        lines.append("*Context retrieved via adaptive HNSW graph traversal.*")
        return "\n".join(lines)
