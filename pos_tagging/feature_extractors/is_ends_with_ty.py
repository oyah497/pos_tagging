from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_ty")
class IsEndsWithTyExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_TY" if (w.endswith('ty') or w.endswith('ties')) else None for w in words]
