"""Agent registry — register and create agents by name."""

from __future__ import annotations

from typing import Any, Callable, Type

from alpha_q.agents.base import BaseAgent

_REGISTRY: dict[str, Type[BaseAgent]] = {}


def register(name: str) -> Callable:
    """Decorator to register an agent class under *name*."""

    def wrapper(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        if name in _REGISTRY:
            raise ValueError(f"Agent '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return wrapper


def create_agent(name: str, **kwargs: Any) -> BaseAgent:
    """Instantiate a registered agent by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown agent '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def available_agents() -> list[str]:
    """Return sorted list of registered agent names."""
    return sorted(_REGISTRY)
