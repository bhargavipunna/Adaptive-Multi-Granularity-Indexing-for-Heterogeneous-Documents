"""
Feature Extraction Engine for AMHIE.
Calculates Visual Dominance Score (VDS), Token Density (Td),
and Layout Complexity (Lc) per document page.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from .parser import ParsedPage, ParsedElement


@dataclass
class PageFeatures:
    """Quantitative multimodal features extracted from a single document page."""
    page_number: int
    page_area: float
    total_words: int
    table_count: int
    image_count: int
    text_block_count: int
    table_area: float
    image_area: float
    visual_dominance_score: float  # VDS in [0.0, 1.0]
    token_density: float           # Td (tokens / cm^2 or scaled)
    layout_complexity: float       # Lc score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "page_area": round(self.page_area, 2),
            "total_words": self.total_words,
            "table_count": self.table_count,
            "image_count": self.image_count,
            "text_block_count": self.text_block_count,
            "table_area": round(self.table_area, 2),
            "image_area": round(self.image_area, 2),
            "vds": round(self.visual_dominance_score, 4),
            "token_density": round(self.token_density, 4),
            "layout_complexity": round(self.layout_complexity, 4)
        }


class FeatureExtractor:
    """
    Computes spatial, textual, and visual density metrics for PDF pages.
    """
    def __init__(self, token_density_scale: float = 10000.0):
        self.token_density_scale = token_density_scale

    @staticmethod
    def _compute_box_area(box: List[float]) -> float:
        if not box or len(box) < 4:
            return 0.0
        x0, y0, x1, y1 = box
        return max(0.0, (x1 - x0) * (y1 - y0))

    def extract_features(self, page: ParsedPage) -> PageFeatures:
        """
        Extracts PageFeatures including VDS, Td, and Lc from a ParsedPage.
        """
        page_area = max(1.0, page.page_area)

        # 1. Table & Image Areas
        table_area = sum(self._compute_box_area(t.bounding_box) for t in page.tables)
        image_area = sum(self._compute_box_area(img.bounding_box) for img in page.images)

        # 2. Visual Dominance Score (VDS)
        # VDS = (Area of Tables + Area of Images) / Total Page Area
        vds = (table_area + image_area) / page_area
        vds = min(1.0, max(0.0, vds))

        # 3. Token Density (Td)
        # Rough token approximation: word count * 1.3
        words = page.full_text.split()
        num_words = len(words)
        approx_tokens = int(num_words * 1.3)
        token_density = (approx_tokens / page_area) * self.token_density_scale

        # 4. Layout Complexity (Lc)
        # Combines block count, spatial dispersion, and multi-column presence
        text_block_count = len(page.text_blocks)
        table_count = len(page.tables)
        image_count = len(page.images)

        # Detect multiple columns by checking distinct horizontal centers of text blocks
        x_centers = []
        for b in page.text_blocks:
            if len(b.bounding_box) >= 4:
                x_center = (b.bounding_box[0] + b.bounding_box[2]) / 2.0
                x_centers.append(x_center)

        distinct_columns = 1
        if len(x_centers) >= 4:
            # Check if columns are distributed across left and right halves
            left_blocks = sum(1 for xc in x_centers if xc < page.width * 0.45)
            right_blocks = sum(1 for xc in x_centers if xc > page.width * 0.55)
            if left_blocks >= 2 and right_blocks >= 2:
                distinct_columns = 2

        # Lc formulation
        lc = (
            (text_block_count * 0.4) +
            (table_count * 2.5) +
            (image_count * 1.5) +
            (distinct_columns - 1) * 3.0
        )

        return PageFeatures(
            page_number=page.page_number,
            page_area=page_area,
            total_words=num_words,
            table_count=table_count,
            image_count=image_count,
            text_block_count=text_block_count,
            table_area=table_area,
            image_area=image_area,
            visual_dominance_score=vds,
            token_density=token_density,
            layout_complexity=lc
        )

    def batch_extract(self, pages: List[ParsedPage]) -> List[PageFeatures]:
        """Extracts features for all pages in a document."""
        return [self.extract_features(p) for p in pages]
