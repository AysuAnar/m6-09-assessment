"""
YOLO26m Inference Application Package.
Exposes the high-performance CatDetector class directly at the package level.
"""

from .detector import CatDetector

__all__ = ["CatDetector"]
