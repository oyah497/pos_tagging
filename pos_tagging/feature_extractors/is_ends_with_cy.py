from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_cy")
class IsEndsWithCyExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_CY" if (w.endswith('cy') or w.endswith('cies'))else None for w in words]
