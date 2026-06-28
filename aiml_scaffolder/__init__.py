"""A guided, dependency-free AI/ML project scaffolder."""

from .generator import ScaffoldResult, scaffold
from .models import ProjectOptions

__all__ = ["ProjectOptions", "ScaffoldResult", "scaffold"]
__version__ = "2.0.0"
