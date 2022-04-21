from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ends_with_ee")
class IsEndsWithEeExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ENDS_WITH_EE" if (w.endswith('ee') or w.endswith('ees')) else None for w in words]
