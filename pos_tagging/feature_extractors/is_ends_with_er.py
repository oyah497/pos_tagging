from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_er")
class IsEndsWithErExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_ER" if (w.endswith('er') or w.endswith('ers')) else None for w in words]
