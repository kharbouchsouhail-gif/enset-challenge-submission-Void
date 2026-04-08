from abc import ABC, abstractmethod
from typing import Any

class BaseTool(ABC):
    """Abstract Base Class for all AI Agent Tools."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Executes the tool's primary function."""
        pass