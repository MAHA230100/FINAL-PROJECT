"""
Feature Registry.
Maps feature names to their implementations.
"""

from typing import Dict, Type
from .core.base_feature import BaseAIFeature


class FeatureRegistry:
    """Registry of all AI features"""
    
    def __init__(self):
        self._features: Dict[str, Type[BaseAIFeature]] = {}
    
    def register(self, name: str, feature_class: Type[BaseAIFeature]):
        """Register a feature"""
        self._features[name] = feature_class
    
    def get(self, name: str) -> Type[BaseAIFeature]:
        """Get a feature class by name"""
        if name not in self._features:
            raise ValueError(f"Unknown feature: {name}")
        return self._features[name]
    
    def list_features(self) -> list:
        """List all registered features"""
        return list(self._features.keys())


# Global registry instance
registry = FeatureRegistry()
