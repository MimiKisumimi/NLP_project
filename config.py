"""
Simulation configuration registry (`config.py`).

Defines the 5 controlled robot-limitation conditions (`LimitationConfig`,
`CONDITION_CONFIGS`), global simulation constants and action durations
(`SimulationConfig`, `CONFIG`), and canonical dish recipes (`DEFAULT_RECIPES`).
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Any, Optional

from models import ActionType, LimitationCondition
from tasks import Recipe, IngredientRequirement


# ---------------------------------------------------------
# Controlled Robot-Limitation Condition Configurations
# ---------------------------------------------------------
@dataclass(frozen=True)
class LimitationConfig:
    """Immutable configuration parameters for one of the 5 robot-limitation conditions."""
    condition: LimitationCondition
    name: str
    description: str
    heterogeneous_capabilities: bool = False          # False = generalist capabilities; True = heterogeneous roles
    partial_observability: bool = False               # False = full visibility of all stations; True = current location only
    sensor_range: Optional[int] = None                # None = full visibility, 1 = current station
    max_messages_per_agent: Optional[int] = None      # Max messages an agent can send per episode (None = unlimited)
    max_comm_tokens_per_agent: Optional[int] = None   # Max communication tokens per agent per episode (None = unlimited)
    memory_limit_k: Optional[int] = None              # Optional sliding-window memory cap (None = unlimited)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this configuration to a dictionary for trace export."""
        return {
            "condition": self.condition.value,
            "name": self.name,
            "description": self.description,
            "heterogeneous_capabilities": self.heterogeneous_capabilities,
            "partial_observability": self.partial_observability,
            "sensor_range": self.sensor_range,
            "max_messages_per_agent": self.max_messages_per_agent,
            "max_comm_tokens_per_agent": self.max_comm_tokens_per_agent,
            "memory_limit_k": self.memory_limit_k,
        }


# Lookup dictionary mapping each `LimitationCondition` enum to its concrete `LimitationConfig`.
CONDITION_CONFIGS: Dict[LimitationCondition, LimitationConfig] = {
    LimitationCondition.REFERENCE: LimitationConfig(
        condition=LimitationCondition.REFERENCE,
        name="Reference",
        description="Full visibility of all stations, broad capabilities, and generous/unlimited communication budget.",
        heterogeneous_capabilities=False,
        partial_observability=False,
        sensor_range=None,
        max_messages_per_agent=None,
        max_comm_tokens_per_agent=None,
        memory_limit_k=None,
    ),
    LimitationCondition.CAPABILITY_LIMITED: LimitationConfig(
        condition=LimitationCondition.CAPABILITY_LIMITED,
        name="Capability-limited",
        description="Robots have heterogeneous action sets, tool compatibilities, and carrying capacities (Carrier, Manipulation, Heat-resistant).",
        heterogeneous_capabilities=True,
        partial_observability=False,
        sensor_range=None,
        max_messages_per_agent=None,
        max_comm_tokens_per_agent=None,
        memory_limit_k=None,
    ),
    LimitationCondition.OBSERVATION_LIMITED: LimitationConfig(
        condition=LimitationCondition.OBSERVATION_LIMITED,
        name="Observation-limited",
        description="Robots see only their current location, local workstation items, and co-located robots.",
        heterogeneous_capabilities=False,
        partial_observability=True,
        sensor_range=1,
        max_messages_per_agent=None,
        max_comm_tokens_per_agent=None,
        memory_limit_k=None,
    ),
    LimitationCondition.COMMUNICATION_LIMITED: LimitationConfig(
        condition=LimitationCondition.COMMUNICATION_LIMITED,
        name="Communication-limited",
        description="Capped maximum number of messages and communication token budget per episode.",
        heterogeneous_capabilities=False,
        partial_observability=False,
        sensor_range=None,
        max_messages_per_agent=8,
        max_comm_tokens_per_agent=200,
        memory_limit_k=None,
    ),
    LimitationCondition.COMBINED: LimitationConfig(
        condition=LimitationCondition.COMBINED,
        name="Combined",
        description="Capability, observation, and communication limitations occur together.",
        heterogeneous_capabilities=True,
        partial_observability=True,
        sensor_range=1,
        max_messages_per_agent=8,
        max_comm_tokens_per_agent=200,
        memory_limit_k=None,
    ),
}


def get_condition_config(condition: Any) -> LimitationConfig:
    """Resolves a condition enum, string name, or integer index (1–5) into a `LimitationConfig`."""
    if isinstance(condition, LimitationConfig):
        return condition
    if isinstance(condition, int):
        int_to_cond = {
            1: LimitationCondition.REFERENCE,
            2: LimitationCondition.CAPABILITY_LIMITED,
            3: LimitationCondition.OBSERVATION_LIMITED,
            4: LimitationCondition.COMMUNICATION_LIMITED,
            5: LimitationCondition.COMBINED,
        }
        if condition in int_to_cond:
            return CONDITION_CONFIGS[int_to_cond[condition]]
    cond_enum = LimitationCondition.from_string(condition)
    if cond_enum and cond_enum in CONDITION_CONFIGS:
        return CONDITION_CONFIGS[cond_enum]
    return CONDITION_CONFIGS[LimitationCondition.CAPABILITY_LIMITED]


# ---------------------------------------------------------
# Centralized Simulation Configuration
# ---------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    """
    Central immutable configuration for physical action durations (in seconds),
    workstation requirements, and action classification sets.
    """
    # Action durations in simulation seconds (determines how long a robot is busy and locks resources)
    action_durations: Dict[ActionType, int] = field(default_factory=lambda: {
        ActionType.MOVE: 1,          # 1 second to move to a target workstation
        ActionType.PICK: 1,          # 1 second to pick up an item
        ActionType.DROP: 1,          # 1 second to drop/place an item
        ActionType.WAIT: 1,          # 1 second idle
        ActionType.CUT: 2,           # 2 seconds
        ActionType.COOK: 2,          # 2 seconds
        ActionType.BOIL: 2,          # 2 seconds
        ActionType.WASH: 2,          # 2 seconds
        ActionType.ASSEMBLE: 2,      # 2 seconds
        ActionType.MESSAGE: 1,       # 1 second communication turn
    })

    # Workstation required for each specialized physical transformation action
    station_requirements: Dict[ActionType, str] = field(default_factory=lambda: {
        ActionType.CUT: "Cutting table",
        ActionType.COOK: "Stove",
        ActionType.BOIL: "Stove",
        ActionType.WASH: "Sink",
        ActionType.ASSEMBLE: "Food assembly table",
    })

    # Basic physical actions executable by any robot without specialized role skills
    basic_actions: Set[ActionType] = field(default_factory=lambda: {
        ActionType.MOVE,
        ActionType.PICK,
        ActionType.DROP,
        ActionType.WAIT,
    })

    # Actions routed purely through the peer messaging system rather than physical station execution
    communication_actions: Set[ActionType] = field(default_factory=lambda: {
        ActionType.MESSAGE,
    })


# Global singleton configuration instance imported across the codebase
CONFIG = SimulationConfig()


# ---------------------------------------------------------
# Standard Default Recipes
# ---------------------------------------------------------
# Dictionary mapping canonical dish names to their `Recipe` specifications,
# used by `TaskBenchmarkSuite` and `Environment._handle_assemble`.
DEFAULT_RECIPES: Dict[str, Recipe] = {
    "salad": Recipe(
        name="salad",
        output_dish="salad",
        required_ingredients=[
            IngredientRequirement("tomato", "cut", 1),
            IngredientRequirement("lettuce", "cut", 1),
            IngredientRequirement("cucumber", "cut", 1),
        ],
        container="bowl",
        target_station="Food assembly table",
        description="Fresh tossed garden salad with cut tomato, cut lettuce, and cut cucumber in a bowl."
    ),
    "sandwich": Recipe(
        name="sandwich",
        output_dish="sandwich",
        required_ingredients=[
            IngredientRequirement("bread", "raw", 1),
            IngredientRequirement("cheese", "cut", 1),
            IngredientRequirement("tomato", "cut", 1),
        ],
        container="plate",
        target_station="Food assembly table",
        description="Sandwich with bread, cut cheese, and cut tomato served on a plate."
    ),
    "fruit_drink": Recipe(
        name="fruit_drink",
        output_dish="fruit_drink",
        required_ingredients=[
            IngredientRequirement("orange", "cut", 1),
        ],
        container="cup",
        target_station="Food assembly table",
        description="Fresh squeezed fruit drink in a cup."
    ),
    "hamburger": Recipe(
        name="hamburger",
        output_dish="hamburger",
        required_ingredients=[
            IngredientRequirement("meat_patty", "cooked", 1),
            IngredientRequirement("burger_bun", "raw", 1),
            IngredientRequirement("lettuce", "cut", 1),
            IngredientRequirement("tomato", "cut", 1),
        ],
        container="plate",
        target_station="Food assembly table",
        description="Juicy hamburger with cooked meat patty, bun, cut lettuce, and cut tomato on a plate."
    ),
    "vegetable_soup": Recipe(
        name="vegetable_soup",
        output_dish="vegetable_soup",
        required_ingredients=[
            IngredientRequirement("carrot", "boiled", 1),
            IngredientRequirement("potato", "boiled", 1),
        ],
        container="bowl",
        target_station="Food assembly table",
        description="Warm vegetable soup made with boiled carrots and potatoes served in a bowl."
    ),
}


__all__ = [
    "LimitationConfig",
    "CONDITION_CONFIGS",
    "get_condition_config",
    "SimulationConfig",
    "CONFIG",
    "DEFAULT_RECIPES",
]
