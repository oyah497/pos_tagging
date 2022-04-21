from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_ance")
class IsEndsWithAnceExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_ANCE" if (w.endswith('ance') or w.endswith('ances')) else None for w in words]
