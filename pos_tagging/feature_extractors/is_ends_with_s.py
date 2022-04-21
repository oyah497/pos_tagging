from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_s")
class IsEndsWithSExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_s" if w.endswith('s') else None for w in words]
