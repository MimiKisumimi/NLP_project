"""
Core data models and schemas (`models.py`).

Defines the speech-act taxonomy (`CommunicationIntent`), limitation conditions
(`LimitationCondition`), action primitives (`ActionType`), discrete kitchen objects
(`Item`), structured LLM decision schema (`RobotAction`), and message envelope (`Message`).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------
# Communication Intent Taxonomy
# ---------------------------------------------------------
class CommunicationIntent(str, Enum):
    """
    Taxonomy of the 8 formal speech-act intents used by robots to classify the
    pragmatic purpose of every natural-language peer message.
    """

    PROPOSAL = "PROPOSAL"  # Proposing a task division or plan
    COMMITMENT = "COMMITMENT"  # Committing to perform an action
    REQUEST = "REQUEST"  # Asking for tools, ingredients, or assistance
    STATUS_UPDATE = (
        "STATUS_UPDATE"  # Reporting milestone, capability, or task completion
    )
    WARNING = "WARNING"  # Alerting peers about collisions or occupied resources
    CONFLICT_RESOLUTION = (
        "CONFLICT_RESOLUTION"  # Resolving disputes over shared resources
    )
    PLAN_REVISION = "PLAN_REVISION"  # Modifying plan after delays or failures
    FAILURE_EXPLANATION = "FAILURE_EXPLANATION"  # Explaining why an action failed

    @classmethod
    def from_string(cls, val: Any) -> Optional["CommunicationIntent"]:
        """Resolves an existing enum or matching string into a CommunicationIntent member."""
        if isinstance(val, cls):
            return val
        if isinstance(val, str):
            cleaned = val.strip().upper()
            try:
                return cls(cleaned)
            except ValueError:
                try:
                    return cls[cleaned]
                except KeyError:
                    return None
        return None


# ---------------------------------------------------------
# Controlled Robot-Limitation Conditions
# ---------------------------------------------------------
class LimitationCondition(str, Enum):
    """Enumeration of the 5 controlled robot-limitation experimental conditions."""

    REFERENCE = "reference"  # Condition 1: Full visibility, broad capabilities, generous comm budget
    CAPABILITY_LIMITED = "capability_limited"  # Condition 2: Heterogeneous action sets & carrying capacities
    OBSERVATION_LIMITED = "observation_limited"  # Condition 3: Partial visibility (current location & nearby objects)
    COMMUNICATION_LIMITED = (
        "communication_limited"  # Condition 4: Capped message count or token budget
    )
    COMBINED = "combined"  # Condition 5: Capability + observation + communication limits

    @classmethod
    def from_string(cls, val: Any) -> Optional["LimitationCondition"]:
        """Resolves an existing enum or matching string into a LimitationCondition member."""
        if isinstance(val, cls):
            return val
        if isinstance(val, str):
            cleaned = val.strip().lower()
            try:
                return cls(cleaned)
            except ValueError:
                try:
                    return cls[val.strip().upper()]
                except KeyError:
                    return None
        return None


# ---------------------------------------------------------
# Action Enum
# ---------------------------------------------------------
class ActionType(str, Enum):
    """Enumeration of all physical navigation, item manipulation, station processing, and communication actions."""

    # Basic physical actions
    MOVE = "MOVE"
    PICK = "PICK"
    DROP = "DROP"
    WAIT = "WAIT"

    # Station tasks
    CUT = "CUT"
    COOK = "COOK"
    BOIL = "BOIL"
    WASH = "WASH"
    ASSEMBLE = "ASSEMBLE"

    # Communication & System actions
    MESSAGE = "MESSAGE"
    ERROR = "ERROR"

    # Backward-compatible alias: MOVE_ITEMS maps directly to PICK
    MOVE_ITEMS = "PICK"

    @classmethod
    def from_string(cls, val: Any) -> Optional["ActionType"]:
        """Resolves an existing enum or matching string into an ActionType member."""
        if isinstance(val, cls):
            return val
        if isinstance(val, str):
            cleaned = val.strip().upper()
            if cleaned == "MOVE_ITEMS":
                return cls.PICK
            try:
                return cls(cleaned)
            except ValueError:
                try:
                    return cls[cleaned]
                except KeyError:
                    return None
        return None


# ---------------------------------------------------------
# Discrete Object & Tool System
# ---------------------------------------------------------
@dataclass
class Item:
    """Represents a discrete ingredient, tool, or finished dish in the kitchen environment."""

    id: str
    name: str
    location: str  # Workstation name or robot_id
    is_tool: bool = False
    state: str = (
        "raw"  # "raw", "cut", "cooked", "boiled", "washed", "assembled", "ready", "served"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `Item` into a dictionary for JSON state snapshots and trace logs."""
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "is_tool": self.is_tool,
            "state": self.state,
        }

    def __str__(self) -> str:
        """Returns a human-readable string showing the item's name and either `(tool)` or its current state."""
        if self.is_tool:
            return f"{self.name} (tool)"
        return f"{self.name} ({self.state})"


# ---------------------------------------------------------
# Pydantic RobotAction Model (Single-Turn: Communicate OR Act)
# ---------------------------------------------------------
class RobotAction(BaseModel):
    """
    Structured LLM response schema for a single robot turn.
    Enforces single-turn decision rules: either send a `MESSAGE` or execute one physical action.
    """

    should_act: bool = Field(
        default=True,
        description="Whether the robot wants to communicate or take a physical action this turn.",
    )

    # --- Action / Function Selection ---
    action: Optional[ActionType] = Field(
        description="Action to execute: MOVE, PICK, DROP, CUT, COOK, BOIL, WASH, ASSEMBLE, MOVE_ITEMS, WAIT, or MESSAGE.",
    )
    function: Optional[ActionType] = Field(
        description="Function type (alias for action): MOVE, PICK, DROP, CUT, COOK, BOIL, WASH, ASSEMBLE, MOVE_ITEMS, WAIT, or MESSAGE.",
    )

    # --- Communication Fields (Used when function == 'MESSAGE') ---
    recipient: str = Field(
        default="ALL",
        description='Recipient of the message when function is MESSAGE. "ALL" for broadcast, or specific robot_id (e.g., "Robot_1").',
    )
    intent: Optional[CommunicationIntent] = Field(
        description="Communication intent (speech act) when function is MESSAGE: PROPOSAL, COMMITMENT, REQUEST, STATUS_UPDATE, WARNING, CONFLICT_RESOLUTION, PLAN_REVISION, FAILURE_EXPLANATION.",
    )
    message_content: str = Field(
        description="The natural language message content to send to peers when function is MESSAGE.",
    )
    content: str = Field(
        default="",
        description="Reasoning or natural language message content (alias for message_content).",
    )

    # --- Physical Action Target Fields ---
    target: Optional[str] = Field(
        default=None,
        description="Unified target field: workstation name (e.g. 'Refrigerator', 'Cutting table') or item name.",
    )
    target_station: Optional[str] = Field(
        default=None,
        description="Target workstation name (e.g. 'Cutting table', 'Refrigerator', 'Sink', 'Stove', 'Food assembly table', 'Serving table') if navigating to a station.",
    )
    target_item: Optional[str] = Field(
        default=None,
        description="Target item, ingredient, or tool for PICK, DROP, CUT, COOK, BOIL, WASH, ASSEMBLE, etc.",
    )

    @field_validator("function", "action", mode="before")
    @classmethod
    def parse_function(cls, v: Any) -> Any:
        """Pydantic pre-validator that coerces raw string action names into `ActionType` enums."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            parsed = ActionType.from_string(v)
            if parsed:
                return parsed
        return v

    @field_validator("intent", mode="before")
    @classmethod
    def parse_intent(cls, v: Any) -> Any:
        """Pydantic pre-validator that coerces raw speech-act strings into `CommunicationIntent` enums."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            parsed = CommunicationIntent.from_string(v)
            if parsed:
                return parsed
        return v

    @field_validator("content", "message_content", mode="before")
    @classmethod
    def parse_content(cls, v: Any) -> str:
        """Pydantic pre-validator that ensures `content` and `message_content` are always non-null strings."""
        if v is None:
            return ""
        return str(v)

    @model_validator(mode="after")
    def normalize_fields(self) -> "RobotAction":
        """Synchronizes aliased fields, sets default communication intent, and unpacks `target`."""
        # 1. Sync action and function
        if self.action is None and self.function is not None:
            self.action = self.function
        elif self.function is None and self.action is not None:
            self.function = self.action
        elif self.action is None and self.function is None:
            self.action = (
                ActionType.MESSAGE
                if (self.message_content or self.intent)
                else ActionType.WAIT
            )
            self.function = self.action

        # 2. Sync content and message_content
        if self.message_content and not self.content:
            self.content = self.message_content
        elif self.content and not self.message_content:
            self.message_content = self.content

        # 3. Strictly enforce Communicate OR Act: only MESSAGE turns carry a communication intent
        if self.function == ActionType.MESSAGE:
            if self.intent is None:
                self.intent = CommunicationIntent.STATUS_UPDATE
        else:
            self.intent = None

        # 4. Migrate unified target into target_station / target_item
        if self.target:
            stripped = self.target.strip()
            if self.function == ActionType.MOVE and not self.target_station:
                self.target_station = stripped
            elif self.function in (
                ActionType.PICK,
                ActionType.DROP,
                ActionType.CUT,
                ActionType.COOK,
                ActionType.BOIL,
                ActionType.WASH,
                ActionType.ASSEMBLE,
            ):
                if not self.target_item:
                    self.target_item = stripped

        # 5. Ensure target attribute is populated if specific target fields were supplied
        if not self.target:
            if self.target_station:
                self.target = self.target_station
            elif self.target_item:
                self.target = self.target_item

        return self


# ---------------------------------------------------------
# Standardized Message Model
# ---------------------------------------------------------
@dataclass
class Message:
    """
    Message dataclass representing a communication message or physical action intention.
    Records sender, recipient, timestep, content, intent (speech act),
    local_state snapshot, and previous_action, alongside physical action targets.
    """

    sender: str  # sender robot_id or "System"
    recipient: str  # "ALL" when broadcasting or specific robot_id
    timestep: int  # simulation second (t)
    function: Union[
        ActionType, str
    ]  # e.g., ActionType.MOVE, ActionType.CUT, ActionType.MESSAGE
    content: str  # text content
    intent: Optional[CommunicationIntent] = None  # Intended communication function
    local_state: Optional[Dict[str, Any]] = None  # Local environment state when sent
    previous_action: Optional[str] = None  # Previous action of sender
    message_content: str = ""  # Peer communication text
    target_station: Optional[str] = None  # Target workstation name
    target_item: Optional[str] = None  # Target discrete item/tool
    target: Optional[str] = None  # Unified target representation

    def __init__(
        self,
        sender: str,
        recipient: str,
        timestep: int,
        function: Union[ActionType, str] = ActionType.MESSAGE,
        content: str = "",
        target_station: Optional[str] = None,
        target_item: Optional[str] = None,
        target: Optional[str] = None,
        intent: Optional[Union[CommunicationIntent, str]] = None,
        local_state: Optional[Dict[str, Any]] = None,
        previous_action: Optional[str] = None,
        message_content: str = "",
    ):
        """
        Initializes a `Message` object, normalizing action types, communication intents,
        and station/item targets so downstream validators and loggers receive consistent types.
        """
        self.sender = sender
        self.recipient = recipient
        self.timestep = timestep

        # Normalize function to ActionType if possible
        parsed_func = ActionType.from_string(function)
        self.function = parsed_func if parsed_func else function

        self.content = content or message_content
        self.message_content = message_content or content

        # Normalize intent to CommunicationIntent if possible
        parsed_intent = (
            CommunicationIntent.from_string(intent) if intent is not None else None
        )
        if parsed_intent is None and isinstance(self.function, ActionType):
            if self.function == ActionType.ERROR:
                parsed_intent = CommunicationIntent.FAILURE_EXPLANATION
            elif self.function == ActionType.MESSAGE:
                parsed_intent = CommunicationIntent.STATUS_UPDATE
        self.intent = parsed_intent

        self.local_state = local_state if local_state is not None else {}
        self.previous_action = previous_action

        self.target_station = target_station
        self.target_item = target_item

        if target and not (self.target_station or self.target_item):
            if self.function == ActionType.MOVE:
                self.target_station = target.strip()
            else:
                self.target_item = target.strip()

        self.target = target or self.target_station or self.target_item or ""

    def get_intent_str(self) -> str:
        """Returns the formatted string representation of the message's communication intent or function."""
        if isinstance(self.intent, CommunicationIntent):
            return self.intent.value
        return str(self.intent) if self.intent else str(self.function)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the message and its metadata into a dictionary for JSON trace export."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "timestep": self.timestep,
            "function": (
                self.function.value
                if isinstance(self.function, ActionType)
                else str(self.function)
            ),
            "intent": (
                self.intent.value
                if isinstance(self.intent, CommunicationIntent)
                else (str(self.intent) if self.intent else None)
            ),
            "content": self.content,
            "message_content": self.message_content,
            "local_state": self.local_state,
            "previous_action": self.previous_action,
            "target": self.target,
            "target_station": self.target_station,
            "target_item": self.target_item,
        }
