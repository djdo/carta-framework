from typing import List, Dict
from .base import BaseStateManager


class MemoryStateManager(BaseStateManager):
    """Ephemeral in-memory session store for a single process."""

    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._state: Dict[str, Dict] = {}

    def get_history(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        history = self._sessions.get(session_id, [])
        return history[-limit:] if limit > 0 else history

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_pedagogical_state(self, session_id: str) -> Dict:
        return self._state.get(session_id, {})

    def update_pedagogical_state(self, session_id: str, state_delta: Dict) -> None:
        self._state.setdefault(session_id, {}).update(state_delta)
