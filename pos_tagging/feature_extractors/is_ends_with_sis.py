from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_sis")
class IsEndsWithSisExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_SIS" if (w.endswith('sis') or w.endswith('ses')) else None for w in words]
