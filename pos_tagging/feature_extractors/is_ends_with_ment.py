from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_ment")
class IsEndsWithMentExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_MENT" if (w.endswith('ment') or w.endswith('ments')) else None for w in words]
