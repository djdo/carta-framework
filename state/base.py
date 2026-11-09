from abc import ABC, abstractmethod
from typing import List, Dict


class BaseStateManager(ABC):
    """Abstract interface decoupling session state storage from chat orchestration."""

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """Retrieve the most recent conversation messages for a session."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a conversational message to the session state."""
        pass

    @abstractmethod
    def clear_history(self, session_id: str) -> None:
        """Reset conversation history for a given session."""
        pass

    # Optional pedagogical state (e.g. hint level reached in a Socratic exchange).
    # Not required by ChatEngine; implementations may override.

    def get_pedagogical_state(self, session_id: str) -> Dict:
        """Retrieve pedagogical state for a session."""
        return {}

    def update_pedagogical_state(self, session_id: str, state_delta: Dict) -> None:
        """Update pedagogical state with a delta dict."""
        raise NotImplementedError(f"{type(self).__name__} does not store pedagogical state")
