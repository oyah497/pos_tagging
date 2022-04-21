from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_tion")
class IsEndsWithTionExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_TION" if (w.endswith('tion') or w.endswith('tions')) else None for w in words]
