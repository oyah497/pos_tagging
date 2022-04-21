from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_sion")
class IsEndsWithsSionExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_SION" if (w.endswith('sion') or w.endswith('sions')) else None for w in words]
