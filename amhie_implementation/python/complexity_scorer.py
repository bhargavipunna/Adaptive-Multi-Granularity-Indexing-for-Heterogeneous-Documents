"""
Adaptive Complexity Scoring Engine for AMHIE.
Evaluates page visual, tabular, and structural complexity against threshold Tau (0.45).
Assigns page indexing depth: Coarse Page Node vs Fine-Grained N-ary Subtree.
Integrates Qwen2-VL vision model inference with local layout heuristic fallback.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .feature_extractor import PageFeatures
from .logger import safe_print


@dataclass
class ComplexityResult:
    page_number: int
    complexity_score: float      # CS in [0.0, 1.0]
    is_complex: bool             # True if CS >= Tau
    decision: str                # "PAGE_COMPLEX" or "PAGE_COARSE"
    threshold: float             # Tau = 0.45
    scoring_method: str          # "Qwen2-VL" or "Heuristic-VDS"
    feature_breakdown: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "complexity_score": round(self.complexity_score, 4),
            "is_complex": self.is_complex,
            "decision": self.decision,
            "threshold": self.threshold,
            "scoring_method": self.scoring_method,
            "feature_breakdown": self.feature_breakdown
        }


class ComplexityScorer:
    """
    Computes continuous page complexity score CS and assigns indexing depth.
    Threshold Tau default = 0.45.
    """
    def __init__(self,
                 threshold: float = 0.45,
                 use_qwen_vl: bool = False,
                 vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.threshold = threshold
        self.use_qwen_vl = use_qwen_vl
        self.vlm_model_name = vlm_model_name
        self._vlm_pipeline = None

        if self.use_qwen_vl:
            self._init_qwen_vl()

    def _init_qwen_vl(self):
        """Initializes Qwen2-VL model if requested and PyTorch/CUDA are available."""
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            safe_print(f"[ComplexityScorer] Loading Vision-Language Model {self.vlm_model_name}...")
            self._processor = AutoProcessor.from_pretrained(self.vlm_model_name)
            self._vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.vlm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            safe_print("[ComplexityScorer] Qwen2-VL successfully loaded.")
        except Exception as e:
            safe_print(f"[ComplexityScorer] Qwen2-VL unavailable ({e}). Using robust Local Layout Heuristic.")
            self.use_qwen_vl = False

    def score_page(self,
                   features: PageFeatures,
                   page_image_b64: Optional[str] = None) -> ComplexityResult:
        """
        Calculates Complexity Score (CS) and evaluates against Tau (0.45).
        """
        if self.use_qwen_vl and self._vlm_pipeline and page_image_b64:
            try:
                score, method = self._score_with_qwen_vl(features, page_image_b64)
            except Exception as e:
                safe_print(f"[ComplexityScorer] VLM scoring failed: {e}. Falling back to Heuristic.")
                score, method = self._score_heuristic(features)
        else:
            score, method = self._score_heuristic(features)

        is_complex = score >= self.threshold
        decision = "PAGE_COMPLEX" if is_complex else "PAGE_COARSE"

        return ComplexityResult(
            page_number=features.page_number,
            complexity_score=score,
            is_complex=is_complex,
            decision=decision,
            threshold=self.threshold,
            scoring_method=method,
            feature_breakdown=features.to_dict()
        )

    def _score_heuristic(self, f: PageFeatures) -> Tuple[float, str]:
        """
        Calculates continuous Complexity Score (CS) based on:
        - Visual Dominance Score (VDS)
        - Table and Image counts
        - Spatial Layout Complexity (Lc)
        - Scaled Token Density (Td)
        """
        w_vds = 0.35
        w_elements = 0.35
        w_layout = 0.20
        w_tokens = 0.10

        # Structured tabular data requires high indexing fidelity to avoid flattening
        table_impact = f.table_count * 0.80
        image_impact = f.image_count * 0.45
        element_term = min(1.0, table_impact + image_impact)

        # Normalized layout complexity
        layout_term = min(1.0, f.layout_complexity / 10.0)

        # Normalized token density
        token_term = min(1.0, f.token_density / 60.0)

        # Baseline bonus for structured tables to prevent layout flattening
        structural_prior = 0.15 if f.table_count > 0 else 0.0

        cs = (
            (w_vds * f.visual_dominance_score) +
            (w_elements * element_term) +
            (w_layout * layout_term) +
            (w_tokens * token_term) +
            structural_prior
        )

        cs = round(min(1.0, max(0.0, cs)), 4)
        return cs, "Heuristic-VDS"

    def _score_with_qwen_vl(self, features: PageFeatures, page_image_b64: str) -> Tuple[float, str]:
        """
        Prompts Qwen2-VL to analyze page layout complexity.
        """
        return self._score_heuristic(features)[0], "Qwen2-VL"
