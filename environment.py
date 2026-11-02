"""
Physical environment and state-transition validation engine (`environment.py`).

Implements the 6-workstation kitchen representation (`Workstation`, `DEFAULT_WORKSTATIONS`),
action completion handlers (`ACTION_COMPLETION_HANDLERS`), physical precondition and lock
validation (`ActionValidator`), and the per-second simulation loop (`Environment`).
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from models import (
    ActionType,
    CommunicationIntent,
    LimitationCondition,
    Item,
    Message,
)
from config import (
    CONFIG,
    DEFAULT_RECIPES,
    LimitationConfig,
    CONDITION_CONFIGS,
    get_condition_config,
)
from tasks import Task, TaskVerifier, TaskVerificationResult
from metrics import (
    FailureTrigger,
    FailureManager,
    MetricsEngine,
    TaskPerformanceMetrics,
    CommunicationMetrics,
)
from observer_judge import PassiveObserver, LLMJudge, LLMJudgeReport


# ---------------------------------------------------------
# Object-Oriented Workstation Representation
# ---------------------------------------------------------
@dataclass
class Workstation:
    """Encapsulates a kitchen workstation and its string aliases."""

    name: str
    aliases: List[str] = field(default_factory=list)
    code: str = ""  # Short display code, e.g. "FRG", "CUT"

    def matches(self, query: str) -> bool:
        """Checks if a string matches the workstation canonical name or any alias."""
        cleaned = query.strip().upper()
        if cleaned == self.name.upper():
            return True
        return any(cleaned == alias.upper() for alias in self.aliases)


DEFAULT_WORKSTATIONS: List[Workstation] = [
    Workstation(
        name="Refrigerator",
        aliases=["FOOD STORAGE", "REFRIGERATOR", "FRIDGE"],
        code="FRG",
    ),
    Workstation(
        name="Cutting table",
        aliases=["CUTTING TABLE", "CUTTING STATION"],
        code="CUT",
    ),
    Workstation(
        name="Sink",
        aliases=["SINK"],
        code="SNK",
    ),
    Workstation(
        name="Serving table",
        aliases=["SERVING TABLE"],
        code="SRV",
    ),
    Workstation(
        name="Stove",
        aliases=["STOVE"],
        code="STV",
    ),
    Workstation(
        name="Food assembly table",
        aliases=["FOOD ASSEMBLY TABLE", "ASSEMBLY STATION", "ASSEMBLY TABLE"],
        code="ASM",
    ),
]


def resolve_workstation(
    name_or_alias: str, workstations: Optional[Dict[str, Workstation]] = None
) -> Optional[Workstation]:
    """Resolves a station name or alias into its canonical `Workstation` instance, or `None`."""
    if not name_or_alias or not isinstance(name_or_alias, str):
        return None
    ws_list = (
        list(workstations.values())
        if workstations is not None
        else DEFAULT_WORKSTATIONS
    )
    for ws in ws_list:
        if ws.matches(name_or_alias):
            return ws
    return None


# ---------------------------------------------------------
# Action Completion Strategy Handlers
# ---------------------------------------------------------
def _handle_pick(env: "Environment", agent: Any, act: Dict[str, Any]):
    """
    Completion handler for `ActionType.PICK`: transfers the target item from the
    current workstation into the robot's inventory if carrying capacity allows.
    """
    target_ws = act.get("station")
    target_item_name = act.get("target_item")
    item = env.find_item_at_station(target_ws, target_item_name) if target_ws else None
    if item and agent.can_carry():
        agent.add_item(item)
        print(f"[Environment] >>> {agent.robot_id} now carries: {item.name}")


def _handle_drop(env: "Environment", agent: Any, act: Dict[str, Any]):
    """
    Completion handler for `ActionType.DROP`: removes target item(s) from the robot's
    inventory and places them at the current workstation (or redirects to a wrong
    station if `FailureManager.intercept_drop_completion` triggers a misplaced delivery).
    Supports dropping a single named item or all held items via target_item='all'.
    Marks assembled dishes placed on the `Serving table` as `'served'`.
    """
    target_ws = act.get("station")
    target_item_name = act.get("target_item")

    if target_item_name and target_item_name.lower() in ("all", "*", "everything"):
        items_to_drop = list(agent.inventory)
        agent.inventory.clear()
    else:
        single = (
            agent.remove_item(target_item_name)
            if target_item_name
            else (agent.inventory.pop(0) if agent.inventory else None)
        )
        items_to_drop = [single] if single else []

    for item in items_to_drop:
        final_ws = target_ws
        if getattr(env, "failure_manager", None):
            overridden_ws = env.failure_manager.intercept_drop_completion(
                env, agent, act, env.current_time
            )
            if overridden_ws:
                final_ws = overridden_ws
        if final_ws:
            item.location = final_ws
            print(f"[Environment] >>> {agent.robot_id} placed {item.name} at {final_ws}")
            if final_ws == "Serving table" and item.state in ("assembled", "ready"):
                item.state = "served"
                print(f"[Environment] >>> {item.name} is now SERVED at {final_ws}!")


# Backward-compatible alias: MOVE_ITEMS is simply PICK
_handle_move_items = _handle_pick


ACTION_STATE_TRANSITIONS: Dict[ActionType, Tuple[str, str]] = {
    ActionType.CUT: ("Cutting table", "cut"),
    ActionType.COOK: ("Stove", "cooked"),
    ActionType.BOIL: ("Stove", "boiled"),
    ActionType.WASH: ("Sink", "washed"),
}


def _handle_state_transition(
    env: "Environment",
    agent: Any,
    act: Dict[str, Any],
    default_station: str,
    target_state: str,
):
    """Generalized state-transition handler for station transformation actions (CUT, COOK, BOIL, WASH)."""
    target_item_name = act.get("target_item")
    station_name = act.get("station") or default_station
    item = agent.get_item(target_item_name) or env.find_item_at_station(
        station_name, target_item_name
    )
    if item:
        item.state = target_state
        print(f"[Environment] >>> {item.name} is now {target_state.upper()}.")


def _make_state_transition_handler(
    action_type: ActionType,
) -> Callable[["Environment", Any, Dict[str, Any]], None]:
    """Factory creating an action completion handler from ACTION_STATE_TRANSITIONS."""
    default_station, target_state = ACTION_STATE_TRANSITIONS[action_type]

    def _handler(env: "Environment", agent: Any, act: Dict[str, Any]):
        _handle_state_transition(env, agent, act, default_station, target_state)

    return _handler


_handle_cut = _make_state_transition_handler(ActionType.CUT)
_handle_cook = _make_state_transition_handler(ActionType.COOK)
_handle_boil = _make_state_transition_handler(ActionType.BOIL)
_handle_wash = _make_state_transition_handler(ActionType.WASH)


def _handle_assemble(env: "Environment", agent: Any, act: Dict[str, Any]):
    """
    Completion handler for `ActionType.ASSEMBLE`: matches the target dish against active
    task recipes or `DEFAULT_RECIPES`, verifies ingredient/container satisfaction at the
    assembly station (plus robot inventory), consumes matched ingredients into the dish,
    and registers a new assembled dish `Item` at the workstation.
    """
    target_ws = act.get("station") or "Food assembly table"
    target_item_name = act.get("target_item")
    target_dish_query = (target_item_name or "").strip()

    candidate_recipes = []
    if env.current_task and env.current_task.recipes:
        candidate_recipes.extend(env.current_task.recipes)
    for r in DEFAULT_RECIPES.values():
        if r not in candidate_recipes:
            candidate_recipes.append(r)

    chosen_recipe = None
    if target_dish_query:
        for r in candidate_recipes:
            if r.matches_target(target_dish_query):
                chosen_recipe = r
                break

    station_items = env.get_items_at_station(target_ws)
    container_items = [it for it in station_items + agent.inventory if it.is_tool]
    all_avail = [it for it in station_items + agent.inventory if not it.is_tool]

    # If the agent did not name a specific recipe, scan candidate recipes and pick the
    # first recipe whose ingredient and container requirements are fully satisfied.
    if not chosen_recipe:
        for r in candidate_recipes:
            sat, _, _, _ = r.check_satisfaction(all_avail, container_items)
            if sat:
                chosen_recipe = r
                break

    if chosen_recipe:
        sat, matched_ingr, matched_cont, reason = chosen_recipe.check_satisfaction(
            all_avail, container_items
        )
        # 1. Enforce ingredient satisfaction
        if not sat:
            # Reject the action if ingredients are missing
            env._reject_physical_action(
                agent, 
                act, 
                ActionType.ASSEMBLE, 
                f"Action ASSEMBLE failed: {reason}", 
                env.current_time
            )
            return False

        dish_name = chosen_recipe.output_dish
        dish_id = f"{dish_name}_{env.item_counter}"
        env.item_counter += 1

        for ingr in matched_ingr:
            if ingr in agent.inventory:
                agent.remove_item(ingr.id)
            if ingr.id in env.items:
                del env.items[ingr.id]

        dish = Item(
            id=dish_id,
            name=dish_name,
            location=target_ws,
            is_tool=False,
            state="assembled",
        )
        env.items[dish_id] = dish
        print(
            f"[Environment] >>> {dish_name} is ASSEMBLED at {dish.location} using recipe '{chosen_recipe.name}'!"
        )
        return True
    else:
        # 2. Reject the action if no matching recipe is found
        env._reject_physical_action(
            agent, 
            act, 
            ActionType.ASSEMBLE, 
            f"Action ASSEMBLE failed: No matching recipe found for '{target_dish_query}'", 
            env.current_time
        )
        return False


ACTION_COMPLETION_HANDLERS: Dict[
    ActionType, Callable[["Environment", Any, Dict[str, Any]], None]
] = {
    ActionType.PICK: _handle_pick,
    ActionType.DROP: _handle_drop,
    ActionType.CUT: _handle_cut,
    ActionType.COOK: _handle_cook,
    ActionType.BOIL: _handle_boil,
    ActionType.WASH: _handle_wash,
    ActionType.ASSEMBLE: _handle_assemble,
}


# ---------------------------------------------------------
# ActionValidator: Encapsulated Physical & Precondition Checks
# ---------------------------------------------------------
class ActionValidator:
    """
    Dedicated validator encapsulating skill checks, state-based station navigation,
    workstation/tool lock validation, and preconditions.
    """

    def __init__(self, workstations: Optional[Dict[str, Workstation]] = None):
        """Initializes the validator with the canonical workstation registry."""
        self.workstations: Dict[str, Workstation] = (
            workstations
            if workstations is not None
            else {ws.name: ws for ws in DEFAULT_WORKSTATIONS}
        )

    def find_workstation(self, name_or_alias: str) -> Optional[Workstation]:
        """Resolves a station name or alias to a `Workstation` object."""
        return resolve_workstation(name_or_alias, self.workstations)

    def validate(
        self,
        agent: Any,
        action: Message,
        environment_state: Dict[str, Any],
        env: Optional[Any] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Validates action and returns `(is_valid, error_msg, target_station_name)`."""
        action_type = ActionType.from_string(action.function)
        func_str = action_type.value if action_type else str(action.function)

        # 1. Skill check
        valid_skill, skill_err = self._validate_skill(agent, action_type, func_str)
        if not valid_skill:
            return False, skill_err, None

        current_loc = environment_state.get("robot_locations", {}).get(
            agent.robot_id, getattr(agent, "location", None)
        )
        if not current_loc:
            return False, f"Agent location unknown for {agent.robot_id}", None

        # 2. WAIT Action
        if action_type == ActionType.WAIT:
            return True, "", str(current_loc)

        # 3. MOVE Action (Direct State-Based Station Assignment)
        if action_type == ActionType.MOVE:
            return self._validate_movement(action, str(current_loc))

        # 4. Workstation and Tool Mutual Exclusion Checks
        current_ws = self.find_workstation(str(current_loc))
        items_dict: Dict[str, Item] = environment_state.get("items", {})
        current_time: int = getattr(env, "current_time", 0) if env else 0

        valid_lock, lock_err = self._validate_workstation_lock(
            agent, action_type, func_str, current_ws, env, current_time
        )
        if not valid_lock:
            return False, lock_err, None

        # 5. PICK Action
        if action_type == ActionType.PICK:
            return self._validate_pick(
                agent,
                action,
                str(current_loc),
                current_ws,
                items_dict,
                env,
                current_time,
            )

        # 6. DROP Action
        if action_type == ActionType.DROP:
            return self._validate_drop(agent, action, str(current_loc), current_ws)

        # 7. Specialized Station Operations (CUT, COOK, BOIL, WASH, ASSEMBLE, MOVE_ITEMS)
        if action_type in CONFIG.station_requirements:
            return self._validate_station_operation(
                agent, action, action_type, str(current_loc), items_dict, env, current_time
            )

        return True, "", str(current_loc)

    def _validate_skill(
        self, agent: Any, action_type: Optional[ActionType], func_str: str
    ) -> Tuple[bool, str]:
        """Verifies that `agent` possesses the required skill for non-basic physical actions."""
        if action_type not in CONFIG.basic_actions and func_str not in agent.skills:
            return (
                False,
                f"Agent lacks required skill: {func_str} (Skills: {agent.skills})",
            )
        return True, ""

    def _validate_movement(
        self,
        action: Message,
        current_loc: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validates a `MOVE` action by resolving `target_station` to one of the canonical
        workstation names.
        """
        station_query = (
            action.target_station or getattr(action, "target", None) or ""
        ).strip()
        workstation = (
            self.find_workstation(station_query) if station_query else None
        )
        if not workstation:
            valid_names = list(self.workstations.keys())
            return (
                False,
                f"Unknown or missing MOVE destination '{station_query}'. Specify target_station from {valid_names}.",
                None,
            )

        if current_loc == workstation.name:
            return (
                False,
                f"Already at '{workstation.name}'. No MOVE needed.",
                None,
            )

        return True, "", workstation.name

    def _validate_workstation_lock(
        self,
        agent: Any,
        action_type: Optional[ActionType],
        func_str: str,
        current_ws: Optional[Workstation],
        env: Optional[Any],
        current_time: int,
    ) -> Tuple[bool, str]:
        """
        Checks whether the current workstation is locked by another robot
        (or by an injected `WORKSTATION_BLOCKED` failure) and logs a resource conflict if contended.
        """
        if current_ws and env and getattr(env, "workstation_locks", None):
            target_ws_name = current_ws.name
            if target_ws_name in env.workstation_locks:
                lock = env.workstation_locks[target_ws_name]
                if (
                    lock["locked_by"] != agent.robot_id
                    and lock["until_timestep"] > current_time
                ):
                    if action_type in (
                        ActionType.CUT,
                        ActionType.COOK,
                        ActionType.BOIL,
                        ActionType.WASH,
                        ActionType.ASSEMBLE,
                        ActionType.PICK,
                        ActionType.DROP,
                    ):
                        env.record_resource_conflict(
                            timestep=current_time,
                            robot_id=agent.robot_id,
                            conflicted_with=lock["locked_by"],
                            resource_type="workstation",
                            resource_name=target_ws_name,
                            attempted_action=func_str,
                        )
                        return (
                            False,
                            f"Resource conflict: Workstation '{target_ws_name}' is currently locked/occupied by {lock['locked_by']} for {lock['action']} until t={lock['until_timestep']}s.",
                        )
        return True, ""

    # Mapping of station actions to required tool names, human-readable requirement label, and conflict label
    ACTION_TOOL_REQUIREMENTS: Dict[ActionType, Tuple[Tuple[str, ...], str, str]] = {
        ActionType.CUT: (("knife",), "a knife", "Knife"),
        ActionType.COOK: (("pan", "pot"), "a pan or pot", "Cookware"),
        ActionType.BOIL: (("pot",), "a pot", "Pot"),
        ActionType.ASSEMBLE: (
            ("bowl", "plate", "cup"),
            "a bowl, plate, or cup",
            "Container",
        ),
    }

    def _validate_tool_lock(
        self,
        agent: Any,
        tool_item: Item,
        attempted_action: str,
        env: Optional[Any],
        current_time: int,
        conflict_label: str = "Tool",
    ) -> Tuple[bool, str]:
        """Validates whether a tool is currently locked by another agent and records resource conflicts."""
        if env and getattr(env, "tool_locks", None) and tool_item.id in env.tool_locks:
            tool_lock = env.tool_locks[tool_item.id]
            if (
                tool_lock["locked_by"] != agent.robot_id
                and tool_lock["until_timestep"] > current_time
            ):
                env.record_resource_conflict(
                    timestep=current_time,
                    robot_id=agent.robot_id,
                    conflicted_with=tool_lock["locked_by"],
                    resource_type="tool",
                    resource_name=tool_item.name,
                    attempted_action=attempted_action,
                )
                return (
                    False,
                    f"Resource conflict: {conflict_label} '{tool_item.name}' ({tool_item.id}) is currently locked/in use "
                    f"by {tool_lock['locked_by']} until t={tool_lock['until_timestep']}s.",
                )
        return True, ""

    def _validate_required_station_tool(
        self,
        agent: Any,
        action_type: ActionType,
        req_station_name: str,
        items_dict: Dict[str, Item],
        env: Optional[Any],
        current_time: int,
    ) -> Tuple[bool, str]:
        """Checks tool availability in inventory or station and validates mutual-exclusion tool locks."""
        if action_type not in self.ACTION_TOOL_REQUIREMENTS:
            return True, ""

        allowed_tools, req_desc, conflict_label = self.ACTION_TOOL_REQUIREMENTS[
            action_type
        ]
        tool_item = next(
            (
                it
                for it in agent.inventory
                if it.is_tool and it.name.lower() in allowed_tools
            ),
            None,
        )
        if not tool_item:
            tool_item = next(
                (
                    it
                    for it in items_dict.values()
                    if it.location == req_station_name
                    and it.is_tool
                    and it.name.lower() in allowed_tools
                ),
                None,
            )
        if not tool_item:
            return (
                False,
                f"Action {action_type.value} requires {req_desc} available at '{req_station_name}' or in robot inventory.",
            )

        return self._validate_tool_lock(
            agent=agent,
            tool_item=tool_item,
            attempted_action=action_type.value,
            env=env,
            current_time=current_time,
            conflict_label=conflict_label,
        )

    def _validate_pick(
        self,
        agent: Any,
        action: Message,
        current_loc: str,
        current_ws: Optional[Workstation],
        items_dict: Dict[str, Item],
        env: Optional[Any],
        current_time: int,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validates a `PICK` action: checks workstation presence, carrying capacity,
        item presence at the station, tool mutual-exclusion locks, and
        robot tool compatibility (`agent.tool_compatibility`).
        """
        if not current_ws:
            return (
                False,
                f"Must be at a workstation to PICK an item. Currently at '{current_loc}'.",
                None,
            )

        if not agent.can_carry():
            return (
                False,
                f"Robot {agent.robot_id} carrying capacity full ({len(agent.inventory)}/{agent.carrying_capacity}). DROP an item first.",
                None,
            )

        target_query = action.target_item or action.content.strip()
        matched_item: Optional[Item] = None
        for it in items_dict.values():
            if it.location == current_ws.name:
                if not target_query or target_query.lower() in (
                    it.name.lower(),
                    it.id.lower(),
                ):
                    matched_item = it
                    break

        if not matched_item:
            return (
                False,
                f"No matching item '{target_query}' found at workstation '{current_ws.name}'.",
                None,
            )

        if matched_item.is_tool:
            valid_tool_lock, tool_lock_err = self._validate_tool_lock(
                agent=agent,
                tool_item=matched_item,
                attempted_action="PICK",
                env=env,
                current_time=current_time,
                conflict_label="Tool",
            )
            if not valid_tool_lock:
                return False, tool_lock_err, None

        if matched_item.is_tool and "all" not in agent.tool_compatibility:
            compatible = any(
                matched_item.name.lower() == t.lower() for t in agent.tool_compatibility
            )
            if not compatible:
                return (
                    False,
                    f"Robot {agent.robot_id} is not compatible with tool '{matched_item.name}'. Compatible tools: {agent.tool_compatibility}.",
                    None,
                )

        return True, "", current_ws.name

    def _validate_drop(
        self,
        agent: Any,
        action: Message,
        current_loc: str,
        current_ws: Optional[Workstation],
    ) -> Tuple[bool, str, Optional[str]]:
        """Validates a `DROP` action: checks workstation presence and verifies the robot holds the item."""
        if not current_ws:
            return (
                False,
                f"Must be at a workstation to DROP an item. Currently at '{current_loc}'.",
                None,
            )

        if not agent.inventory:
            return (
                False,
                f"Robot {agent.robot_id} inventory is empty. Nothing to DROP.",
                None,
            )

        target_query = (action.target_item or action.content.strip()).strip()
        if target_query.lower() in ("all", "*", "everything"):
            return True, "", current_ws.name

        item_to_drop = (
            agent.get_item(target_query) if target_query else agent.inventory[0]
        )
        if not item_to_drop:
            inv_names = [it.name for it in agent.inventory]
            return (
                False,
                f"Robot {agent.robot_id} is not carrying '{target_query}'. Current inventory: {inv_names}.",
                None,
            )

        return True, "", current_ws.name

    def _validate_station_operation(
        self,
        agent: Any,
        action: Message,
        action_type: ActionType,
        current_loc: str,
        items_dict: Dict[str, Item],
        env: Optional[Any],
        current_time: int,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validates specialized station operations (`CUT`, `COOK`, `BOIL`, `WASH`, `ASSEMBLE`):
        verifies the robot is at the required station (`CONFIG.station_requirements`),
        checks required tool availability/locks, and enforces action-specific ingredient preconditions.
        """
        req_station_name = CONFIG.station_requirements[action_type]
        current_ws = self.find_workstation(current_loc)
        if not current_ws or current_ws.name != req_station_name:
            return (
                False,
                f"Must be at '{req_station_name}' to perform {action_type.value}. Currently at '{current_loc}'.",
                None,
            )

        # Shared tool presence & lock validation for CUT, COOK, BOIL, and ASSEMBLE
        valid_tool, tool_err = self._validate_required_station_tool(
            agent=agent,
            action_type=action_type,
            req_station_name=req_station_name,
            items_dict=items_dict,
            env=env,
            current_time=current_time,
        )
        if not valid_tool:
            return False, tool_err, None

        target_query = action.target_item or action.content.strip()

        # CUT Preconditions
        if action_type == ActionType.CUT:
            target_item = agent.get_item(target_query) or next(
                (
                    it
                    for it in items_dict.values()
                    if it.location == req_station_name
                    and not it.is_tool
                    and (
                        not target_query
                        or target_query.lower() in (it.name.lower(), it.id.lower())
                    )
                ),
                None,
            )
            if not target_item:
                return (
                    False,
                    f"No cuttable ingredient found at '{req_station_name}' matching '{target_query or 'any'}'.",
                    None,
                )
            if any(
                uncut in target_item.name.lower()
                for uncut in ("meat_patty", "patty", "burger_bun", "bun", "bread")
            ):
                return (
                    False,
                    f"Invalid action: Item '{target_item.name}' cannot be CUT. Meat patties are COOKED at the Stove; buns and bread remain raw.",
                    None,
                )
            if target_item.state == "cut":
                return False, f"Item '{target_item.name}' is already cut.", None

        # COOK Preconditions
        elif action_type == ActionType.COOK:
            target_item = agent.get_item(target_query) or next(
                (
                    it
                    for it in items_dict.values()
                    if it.location == req_station_name
                    and not it.is_tool
                    and (
                        not target_query
                        or target_query.lower() in (it.name.lower(), it.id.lower())
                    )
                ),
                None,
            )
            if not target_item:
                return (
                    False,
                    f"No cookable ingredient found at '{req_station_name}' matching '{target_query or 'any'}'.",
                    None,
                )
            if not any(k in target_item.name.lower() for k in ("meat_patty", "patty")):
                return (
                    False,
                    f"Invalid action: Cannot COOK '{target_item.name}'. Only meat patties are COOKED at the Stove (vegetables for soup are BOILED).",
                    None,
                )
            if target_item.state == "cooked":
                return False, f"Item '{target_item.name}' is already cooked.", None

        # BOIL Preconditions
        elif action_type == ActionType.BOIL:
            target_item = agent.get_item(target_query) or next(
                (
                    it
                    for it in items_dict.values()
                    if it.location == req_station_name
                    and not it.is_tool
                    and (
                        not target_query
                        or target_query.lower() in (it.name.lower(), it.id.lower())
                    )
                ),
                None,
            )
            if not target_item:
                return (
                    False,
                    f"No boilable ingredient found at '{req_station_name}' matching '{target_query or 'any'}'.",
                    None,
                )
            if not any(k in target_item.name.lower() for k in ("carrot", "potato")):
                return (
                    False,
                    f"Invalid action: Cannot BOIL '{target_item.name}'. Only soup vegetables (carrot, potato) can be BOILED.",
                    None,
                )
            if target_item.state == "boiled":
                return False, f"Item '{target_item.name}' is already boiled.", None

        # WASH Preconditions
        elif action_type == ActionType.WASH:
            item_at_sink = agent.get_item(target_query) or next(
                (
                    it
                    for it in items_dict.values()
                    if it.location == req_station_name
                    and (
                        not target_query
                        or target_query.lower() in (it.name.lower(), it.id.lower())
                    )
                ),
                None,
            )
            if not item_at_sink:
                return (
                    False,
                    f"No item found at '{req_station_name}' matching '{target_query or 'any'}' to WASH.",
                    None,
                )
            if any(
                uncut in item_at_sink.name.lower()
                for uncut in ("meat_patty", "patty", "burger_bun", "bun", "bread")
            ):
                return (
                    False,
                    f"Invalid action: Cannot WASH '{item_at_sink.name}'. Only fruits and vegetables can be WASHED.",
                    None,
                )

        # ASSEMBLE Preconditions
        elif action_type == ActionType.ASSEMBLE:
            prepared_ingredients = [
                it
                for it in items_dict.values()
                if (it.location == req_station_name or agent.has_item(it.id))
                and not it.is_tool
            ]
            if not prepared_ingredients:
                return (
                    False,
                    f"Action ASSEMBLE requires ingredients at '{req_station_name}'.",
                    None,
                )

        return True, "", req_station_name


# ---------------------------------------------------------
# Simulation Environment
# ---------------------------------------------------------
class Environment:
    """
    Central world simulator orchestrating the state-based kitchen workstations, registered `RobotAgent`
    instances, discrete `Item` state transitions, mutual-exclusion locks, `FailureManager`
    injections, `PassiveObserver` trace recording, and `LLMJudge` post-episode evaluation.
    """

    def __init__(
        self,
        task: Optional[Task] = None,
        initial_items: Optional[List[Item]] = None,
        condition: Union[
            LimitationCondition, LimitationConfig, str, int
        ] = LimitationCondition.CAPABILITY_LIMITED,
        failure_triggers: Optional[List[FailureTrigger]] = None,
        failure_seed: int = 42,
        observer: Optional[PassiveObserver] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initializes the simulation environment with a benchmark `Task`, a controlled
        `LimitationCondition`, optional `FailureTrigger` injections, and a `PassiveObserver`.
        """
        self.workstations: Dict[str, Workstation] = {
            ws.name: ws for ws in DEFAULT_WORKSTATIONS
        }
        self.validator = ActionValidator(self.workstations)
        self.action_handlers = ACTION_COMPLETION_HANDLERS

        self.current_time: int = 0
        self.agents: Dict[str, Any] = {}
        self.item_counter: int = 100
        self._task_broadcasted: bool = False

        # Passive Observer: records execution trace without influencing agents
        self.observer: PassiveObserver = observer or PassiveObserver(run_id=run_id)
        self._observer_initialized: bool = False

        # Controlled Robot-Limitation Condition
        self.condition_config: LimitationConfig = get_condition_config(condition)
        self.condition: LimitationCondition = self.condition_config.condition

        # Seed-Controlled Failure Manager
        self.failure_manager: FailureManager = FailureManager(
            triggers=failure_triggers, seed=failure_seed
        )

        # Shared Resource Locks
        self.workstation_locks: Dict[str, Dict[str, Any]] = {}
        self.tool_locks: Dict[str, Dict[str, Any]] = {}
        self.resource_conflicts: List[Dict[str, Any]] = []

        # Current benchmark task
        if task is not None:
            self.current_task = task
        else:
            from tasks import TaskBenchmarkSuite

            self.current_task = TaskBenchmarkSuite.task_1_prepare_salad()

        # Initialize items: from task or passed initial_items or defaults
        if self.current_task and initial_items is None:
            self.items: Dict[str, Item] = {
                it.id: it for it in self.current_task.initial_items
            }
        elif initial_items is not None:
            self.items: Dict[str, Item] = {it.id: it for it in initial_items}
        else:
            default_items = [
                Item(
                    id="knife_1",
                    name="knife",
                    location="Cutting table",
                    is_tool=True,
                    state="ready",
                ),
                Item(
                    id="pan_1",
                    name="pan",
                    location="Stove",
                    is_tool=True,
                    state="ready",
                ),
                Item(
                    id="pot_1",
                    name="pot",
                    location="Stove",
                    is_tool=True,
                    state="ready",
                ),
                Item(
                    id="bowl_1",
                    name="bowl",
                    location="Food assembly table",
                    is_tool=True,
                    state="ready",
                ),
                Item(
                    id="plate_1",
                    name="plate",
                    location="Food assembly table",
                    is_tool=True,
                    state="ready",
                ),
                Item(
                    id="tomato_1",
                    name="tomato",
                    location="Refrigerator",
                    is_tool=False,
                    state="raw",
                ),
                Item(
                    id="lettuce_1",
                    name="lettuce",
                    location="Refrigerator",
                    is_tool=False,
                    state="raw",
                ),
                Item(
                    id="cucumber_1",
                    name="cucumber",
                    location="Refrigerator",
                    is_tool=False,
                    state="raw",
                ),
            ]
            self.items = {it.id: it for it in default_items}

        self.global_state: Dict[str, Any] = {
            "tasks": [self.current_task.to_dict()] if self.current_task else [],
            "condition": self.condition_config.to_dict(),
            "objects": {"task_status": "in_progress"},
            "workstations": list(self.workstations.keys()),
            "robot_locations": {},
            "items": self.items,
            "workstation_locks": self.workstation_locks,
            "tool_locks": self.tool_locks,
            "resource_conflicts": self.resource_conflicts,
        }
        self.message_log: List[Message] = []
        self.peer_message_log: List[Message] = []
        self.completed_actions: List[Dict[str, Any]] = []
        self.failed_actions: List[Dict[str, Any]] = []
        self.state_history: List[Dict[str, Any]] = []

    def find_workstation(self, name_or_alias: str) -> Optional[Workstation]:
        """Finds a workstation by canonical name or alias."""
        return resolve_workstation(name_or_alias, self.workstations)

    def set_condition(
        self,
        condition: Union[LimitationCondition, LimitationConfig, str, int],
        reconfigure_agents: bool = True,
    ):
        """Updates the active limitation condition and optionally reconfigures registered agents."""
        self.condition_config = get_condition_config(condition)
        self.condition = self.condition_config.condition
        self.global_state["condition"] = self.condition_config.to_dict()
        if reconfigure_agents and self.agents:
            from agent import get_agent_profile

            for r_id, agent in self.agents.items():
                prof = get_agent_profile(r_id, self.condition_config)
                agent.role = prof["role"]
                agent.skills = [
                    s.value if isinstance(s, ActionType) else str(s)
                    for s in prof["skills"]
                ]
                agent.carrying_capacity = prof["carrying_capacity"]
                agent.tool_compatibility = prof["tool_compatibility"]
                agent.role_description = prof["description"]
                agent.memory_limit = prof["memory_limit"]
                agent.max_messages = prof["max_messages"]
                agent.max_comm_tokens = prof["max_comm_tokens"]
        print(
            f"[Environment] Active Condition set to: {self.condition_config.name} ({self.condition.value})"
        )

    def broadcast_task_at_start(self):
        """
        Broadcasts the user's task to all agents at the beginning of the run (t=0)
        without providing a global allocation plan.
        """
        if not self.current_task or self._task_broadcasted:
            return
        self._task_broadcasted = True
        if self.observer and not self._observer_initialized:
            self.observer.observe_initialization(self)
            self._observer_initialized = True
        dishes_str = ", ".join(self.current_task.target_dishes)
        announcement = (
            f"TASK BROADCAST: {self.current_task.name} ({self.current_task.difficulty}). "
            f"{self.current_task.description} "
            f"Required final dish(es): [{dishes_str}] delivered to {self.current_task.target_station}. "
            f"Coordinate directly with your peers to discover capabilities and allocate subtasks."
        )
        msg = Message(
            sender="System",
            recipient="ALL",
            timestep=0,
            function=ActionType.MESSAGE,
            intent=CommunicationIntent.STATUS_UPDATE,
            content=announcement,
            previous_action="INIT",
        )
        self.broadcast_message(msg)

    def load_task(self, task: Task):
        """Loads a benchmark task, populating initial items, recipes, DAG, and resetting locks."""
        self.current_task = task
        self.items = {it.id: it for it in task.initial_items}
        self.workstation_locks.clear()
        self.tool_locks.clear()
        self.resource_conflicts.clear()
        self.global_state["tasks"] = [task.to_dict()]
        self.global_state["items"] = self.items
        print(
            f"[Environment] Loaded task: {task.name} ({task.difficulty}) - Target: {task.target_dishes} at {task.target_station}"
        )

    def record_resource_conflict(
        self,
        timestep: int,
        robot_id: str,
        conflicted_with: str,
        resource_type: str,
        resource_name: str,
        attempted_action: str,
    ):
        """Records a mutual exclusion contention event when two agents contest a resource."""
        conflict_event = {
            "timestep": timestep,
            "robot_id": robot_id,
            "conflicted_with": conflicted_with,
            "resource_type": resource_type,
            "resource_name": resource_name,
            "attempted_action": attempted_action,
        }
        self.resource_conflicts.append(conflict_event)
        print(
            f"[Environment] ⚠️ RESOURCE CONFLICT at t={timestep}s: {robot_id} attempted {attempted_action} on {resource_type} '{resource_name}' locked by {conflicted_with}"
        )

    def verify_current_task(self) -> TaskVerificationResult:
        """Ground-truth task verification against deterministic simulator state."""
        return TaskVerifier.verify(self, self.current_task)

    def is_task_complete(self) -> bool:
        """Returns True if the current benchmark task is verified complete."""
        return self.verify_current_task().is_complete

    def sync_task_dag(self):
        """
        Deterministically evaluates ground-truth simulator state to mark completed DAG nodes.
        Passes current items, robot workstation locations, and in-transit held inventories
        to `Task.sync_dag()`, updating task milestone progress in `self.global_state`.
        """
        if not self.current_task:
            return

        items_list = list(self.items.values())
        robot_locs = dict(self.global_state.get("robot_locations", {}))
        robot_invs = {
            r_id: list(agent.inventory)
            for r_id, agent in self.agents.items()
        }
        self.current_task.sync_dag(
            items=items_list,
            robot_locations=robot_locs,
            robot_inventories=robot_invs,
        )

        if self.global_state.get("tasks"):
            self.global_state["tasks"][0]["progress"] = self.current_task.get_progress()

    def _find_required_tool_for_action(
        self, action_type: ActionType, station_name: str, agent: Any
    ) -> Optional[Item]:
        """
        Finds the tool Item required by the agent for a specialized station action.
        Searches first the robot's held inventory and then items resting at the workstation counter.
        """
        if action_type not in ActionValidator.ACTION_TOOL_REQUIREMENTS:
            return None

        allowed_tools, _, _ = ActionValidator.ACTION_TOOL_REQUIREMENTS[action_type]
        for tool_name in allowed_tools:
            for it in agent.inventory:
                if it.is_tool and it.name.lower() == tool_name:
                    return it
            for it in self.get_items_at_station(station_name):
                if it.is_tool and it.name.lower() == tool_name:
                    return it
        return None

    def get_items_at_station(self, station_name: str) -> List[Item]:
        """Returns all items currently resting on a workstation's counter."""
        ws = self.find_workstation(station_name)
        target_name = ws.name if ws else station_name
        return [it for it in self.items.values() if it.location == target_name]

    def find_item_at_station(
        self, station_name: str, item_name_or_id: Optional[str] = None
    ) -> Optional[Item]:
        """Finds a matching item at a given workstation."""
        station_items = self.get_items_at_station(station_name)
        if not item_name_or_id:
            return station_items[0] if station_items else None
        query = item_name_or_id.strip().lower()
        for it in station_items:
            if query in (it.name.lower(), it.id.lower()):
                return it
        return None

    def add_agent(self, agent: Any, starting_location: Any = "Refrigerator"):
        """
        Registers a `RobotAgent` in the environment at a workstation string `starting_location`
        and registers any held items.
        """
        ws = (
            self.find_workstation(starting_location)
            if isinstance(starting_location, str)
            else None
        )
        loc_str = ws.name if ws else "Refrigerator"

        agent.location = loc_str
        self.agents[agent.robot_id] = agent
        self.global_state["robot_locations"][agent.robot_id] = loc_str
        for it in agent.inventory:
            it.location = agent.robot_id
            self.items[it.id] = it

    def broadcast_message(self, message: Message):
        """Delivers a broadcast message to all agents except the sender."""
        intent_tag = (
            f" [{message.intent.value}]"
            if isinstance(message.intent, CommunicationIntent)
            else (f" [{message.intent}]" if message.intent else "")
        )
        print(
            f"[Environment] Broadcasting message from {message.sender}{intent_tag}: {message.content}"
        )
        self.message_log.append(message)
        if message.sender != "System":
            self.peer_message_log.append(message)
        for robot_id, agent in self.agents.items():
            if robot_id != message.sender:
                agent.receive_message(message)

    def _deliver_routed_message(self, message: Message) -> bool:
        """Internal helper to deliver a direct or broadcast message to recipient(s)."""
        if message.recipient.upper() == "ALL":
            self.broadcast_message(message)
            return True
        elif message.recipient in self.agents:
            intent_tag = (
                f" [{message.intent.value}]"
                if isinstance(message.intent, CommunicationIntent)
                else (f" [{message.intent}]" if message.intent else "")
            )
            print(
                f"[Environment] Routing direct message from {message.sender} to {message.recipient}{intent_tag}: {message.content}"
            )
            self.message_log.append(message)
            if message.sender != "System":
                self.peer_message_log.append(message)
            self.agents[message.recipient].receive_message(message)
            return True
        else:
            print(f"[Environment] Error: Recipient {message.recipient} not found.")
            return False

    def route_message(self, message: Message) -> bool:
        """
        Handles routing of messages (direct or broadcast), enforcing communication
        limits under Condition 4 and Condition 5, enriching message state metadata, and
        triggering FailureManager interception for message delay/drop scenarios.
        Returns True if the message was delivered (or queued), False if blocked/invalid.
        """
        # Enforce communication budget for non-System senders
        if message.sender in self.agents:
            sender_agent = self.agents[message.sender]
            tok_count = len(message.message_content.split())
            can_send, budget_err = sender_agent.can_communicate(extra_tokens=tok_count)
            if not can_send:
                print(
                    f"[Environment] ⚠️ Communication blocked for {message.sender}: {budget_err}"
                )
                self.failed_actions.append(
                    {
                        "timestep": message.timestep,
                        "robot_id": message.sender,
                        "function": "MESSAGE",
                        "error": f"Communication limit exceeded: {budget_err}",
                    }
                )
                sender_agent.receive_message(
                    Message(
                        sender="System",
                        recipient=message.sender,
                        timestep=message.timestep,
                        function=ActionType.ERROR,
                        intent=CommunicationIntent.WARNING,
                        content=f"Message blocked: {budget_err}",
                    )
                )
                return False

            # Populate local_state and previous_action on the message if not already set
            if not message.local_state:
                message.local_state = dict(sender_agent.local_observation)
            if not message.previous_action:
                message.previous_action = getattr(
                    sender_agent, "last_action_str", "NONE"
                )
            if message.intent is None:
                message.intent = CommunicationIntent.STATUS_UPDATE

            sender_agent.record_sent_message(message)

            # Check if FailureManager intercepts this message (delayed or omitted message)
            if self.failure_manager and self.failure_manager.intercept_message(
                self, message, message.timestep
            ):
                return True

        return self._deliver_routed_message(message)

    def snapshot_state(self, timestep: int) -> Dict[str, Any]:
        """Stores the complete state snapshot after every timestep."""
        snapshot = {
            "timestep": timestep,
            "condition": self.condition.value,
            "robot_locations": dict(self.global_state["robot_locations"]),
            "robot_inventories": {
                r_id: [item.to_dict() for item in agent.inventory]
                for r_id, agent in self.agents.items()
            },
            "items": {item_id: item.to_dict() for item_id, item in self.items.items()},
            "workstation_items": {
                ws_name: [item.id for item in self.get_items_at_station(ws_name)]
                for ws_name in self.workstations.keys()
            },
            "completed_actions": list(self.completed_actions),
            "failed_actions": list(self.failed_actions),
            "workstation_locks": dict(self.workstation_locks),
            "tool_locks": dict(self.tool_locks),
            "resource_conflicts": list(self.resource_conflicts),
            "task_progress": (
                self.current_task.get_progress() if self.current_task else {}
            ),
            "task_state": json.loads(json.dumps(self.global_state["tasks"])),
        }
        self.state_history.append(snapshot)
        return snapshot

    def _expire_locks(self, timestep: int):
        """Expires workstation and tool locks whose duration has elapsed."""
        self.workstation_locks = {
            ws: lk
            for ws, lk in self.workstation_locks.items()
            if lk["until_timestep"] > timestep
        }
        self.tool_locks = {
            tid: lk
            for tid, lk in self.tool_locks.items()
            if lk["until_timestep"] > timestep
        }

    def _process_completed_actions(self, timestep: int):
        """Finalizes in-progress agent actions that complete at `timestep` via Strategy Pattern handlers."""
        for agent in self.agents.values():
            if agent.current_action and agent.busy_until == timestep:
                act = agent.current_action
                func = act["function"]
                func_type = ActionType.from_string(func)
                print(
                    f"[Environment] >>> {agent.robot_id} COMPLETED {func} (duration: {act['duration']}s)"
                )
                agent.last_action_str = f"{func}({act.get('target_item') or act.get('station') or act.get('target') or ''})"

                # Release locks held by this agent
                stations_to_free = [
                    ws
                    for ws, lk in self.workstation_locks.items()
                    if lk["locked_by"] == agent.robot_id
                ]
                for ws in stations_to_free:
                    del self.workstation_locks[ws]
                tools_to_free = [
                    tid
                    for tid, lk in self.tool_locks.items()
                    if lk["locked_by"] == agent.robot_id
                ]
                for tid in tools_to_free:
                    del self.tool_locks[tid]

                # Strategy Pattern dispatch
                handler_ok = True
                if func_type and func_type in self.action_handlers:
                    handler = self.action_handlers[func_type]
                    res = handler(self, agent, act)
                    if res is False:
                        handler_ok = False

                # Sync DAG progress and evaluate ground-truth completion
                self.sync_task_dag()
                if self.is_task_complete():
                    print(
                        f"[Environment] 🎉 TASK '{self.current_task.name if self.current_task else 'Current'}' IS COMPLETED!"
                    )

                if handler_ok:
                    self.completed_actions.append(
                        {"timestep": timestep, "robot_id": agent.robot_id, "action": act}
                    )
                agent.current_action = None

    def _render_timestep_status(self, timestep: int):
        """Renders the state-based kitchen workstation overview, active resource locks, and robot inventories."""
        if self.current_task:
            prog = self.current_task.get_progress()
            print(
                f"\nTask: {self.current_task.name} ({self.current_task.difficulty}) | Progress: {prog['completed_nodes']}/{prog['total_nodes']} steps ({prog['percent']}%) | Target: {self.current_task.target_dishes} -> {self.current_task.target_station}"
            )

        print("--- Kitchen Workstations (Robots & Items) ---")
        for ws_name in self.workstations:
            robots_here = [
                r_id
                for r_id, loc in self.global_state["robot_locations"].items()
                if loc == ws_name
            ]
            robots_str = f"Robots: {robots_here}" if robots_here else "Robots: []"
            st_items = self.get_items_at_station(ws_name)
            st_str = ", ".join([str(it) for it in st_items]) if st_items else "(empty)"
            print(f"  [{ws_name:^20}] {robots_str:28} | Items: {st_str}")

        if self.workstation_locks or self.tool_locks:
            print("\n--- Active Resource Locks ---")
            for ws, lk in self.workstation_locks.items():
                print(
                    f"  🔒 Station: [{ws}] locked by {lk['locked_by']} ({lk['action']}) until t={lk['until_timestep']}s"
                )
            for tid, lk in self.tool_locks.items():
                print(
                    f"  🔒 Tool: [{lk['tool_name']} ({tid})] locked by {lk['locked_by']} ({lk['action']}) until t={lk['until_timestep']}s"
                )

        print("\n--- Robot Locations, Inventories & Capacities ---")
        for r_id, ag in self.agents.items():
            loc = self.global_state["robot_locations"].get(r_id, ag.location)
            inv_str = (
                ", ".join([str(it) for it in ag.inventory])
                if ag.inventory
                else "(empty)"
            )
            print(
                f"  [{r_id} ({ag.role})] Location: {loc:20} | Capacity: {len(ag.inventory)}/{ag.carrying_capacity} | Items: [{inv_str}]"
            )

    def _update_agent_observations(self, timestep: int):
        """
        Updates each agent's local observation dictionary each timestep.
        Enforces partial observability under Conditions 3 & 5 (masking unobserved workstation
        items, restricting visible peer locations to co-located robots, and scoping resource locks)
        or full observability under Conditions 1, 2, and 4.
        """
        partial_obs = self.condition_config.partial_observability

        for agent in self.agents.values():
            my_loc = str(
                self.global_state["robot_locations"].get(agent.robot_id, agent.location)
            )
            agent.location = my_loc

            if partial_obs:
                # Partial observability: robot sees only its current location and co-located objects/robots
                other_locs = {
                    r_id: loc
                    for r_id, loc in self.global_state["robot_locations"].items()
                    if r_id != agent.robot_id and loc == my_loc
                }
                ws_item_summary = {
                    ws_name: (
                        [str(it) for it in self.get_items_at_station(ws_name)]
                        if ws_name == my_loc
                        else ["(unobserved - at another station)"]
                    )
                    for ws_name in self.workstations
                }
                active_ws_locks = {
                    ws: lk["locked_by"]
                    for ws, lk in self.workstation_locks.items()
                    if lk["until_timestep"] > timestep and ws == my_loc
                }
                active_tool_locks = {
                    lk["tool_name"]: lk["locked_by"]
                    for tid, lk in self.tool_locks.items()
                    if lk["until_timestep"] > timestep
                    and self.items.get(tid)
                    and self.items[tid].location == my_loc
                }
                obs_mode = "partial"
            else:
                other_locs = {
                    r_id: loc
                    for r_id, loc in self.global_state["robot_locations"].items()
                    if r_id != agent.robot_id
                }
                ws_item_summary = {
                    ws_name: [str(it) for it in self.get_items_at_station(ws_name)]
                    for ws_name in self.workstations
                }
                active_ws_locks = {
                    ws: lk["locked_by"]
                    for ws, lk in self.workstation_locks.items()
                    if lk["until_timestep"] > timestep
                }
                active_tool_locks = {
                    lk["tool_name"]: lk["locked_by"]
                    for lk in self.tool_locks.values()
                    if lk["until_timestep"] > timestep
                }
                obs_mode = "full"

            agent.update_observation(
                {
                    "current_time_seconds": timestep,
                    "observability_mode": obs_mode,
                    "condition": self.condition.value,
                    "my_location": my_loc,
                    "is_busy": agent.is_busy(timestep),
                    "busy_remaining_seconds": max(0, agent.busy_until - timestep),
                    "current_action": (
                        agent.current_action["function"]
                        if agent.current_action
                        else None
                    ),
                    "inventory": [it.to_dict() for it in agent.inventory],
                    "carrying_capacity": agent.carrying_capacity,
                    "workstation_items": ws_item_summary,
                    "workstation_locks": active_ws_locks,
                    "tool_locks": active_tool_locks,
                    "task_name": (
                        self.current_task.name if self.current_task else "Task"
                    ),
                    "task_description": (
                        self.current_task.description if self.current_task else ""
                    ),
                    "target_dishes": (
                        self.current_task.target_dishes if self.current_task else []
                    ),
                    "target_station": (
                        self.current_task.target_station
                        if self.current_task
                        else "Serving table"
                    ),
                    "task_progress": {},
                    "other_robot_locations": other_locs,
                }
            )

    def _collect_agent_decisions(self, timestep: int) -> List[Tuple[Any, Message]]:
        """Queries all non-busy agents for their next communication OR physical action."""
        actions: List[Tuple[Any, Message]] = []
        for agent in self.agents.values():
            if agent.is_busy(timestep):
                rem = agent.busy_until - timestep
                func = (
                    agent.current_action["function"]
                    if agent.current_action
                    else "action"
                )
                print(
                    f"[{agent.robot_id}] Busy performing {func} ({rem}s remaining)..."
                )
                continue

            action = agent.reason_and_act(timestep)
            if action is not None:
                actions.append((agent, action))
        return actions

    def _start_valid_physical_action(
        self,
        agent: Any,
        action: Message,
        action_type: Optional[ActionType],
        target_station_name: Optional[str],
        timestep: int,
    ):
        """Initiates a validated physical action (MOVE, WAIT, or station operation) and acquires resource locks."""
        duration = CONFIG.action_durations.get(action_type, 1)

        if action_type == ActionType.MOVE:
            old_loc = self.global_state["robot_locations"].get(
                agent.robot_id, agent.location
            )
            new_loc = target_station_name or old_loc
            agent.location = new_loc
            self.global_state["robot_locations"][agent.robot_id] = new_loc
            agent.busy_until = timestep + duration
            agent.current_action = {
                "function": ActionType.MOVE.value,
                "target": new_loc,
                "station": new_loc,
                "duration": duration,
                "start_time": timestep,
                "finish_time": timestep + duration,
            }
            print(
                f"[Environment] {agent.robot_id} MOVED from '{old_loc}' to '{new_loc}' (takes {duration}s)"
            )
            return

        if action_type == ActionType.WAIT:
            agent.busy_until = timestep + duration
            agent.current_action = {
                "function": ActionType.WAIT.value,
                "duration": duration,
                "start_time": timestep,
                "finish_time": timestep + duration,
            }
            print(f"[Environment] {agent.robot_id} is WAITING for {duration}s.")
            return

        func_name = action_type.value if action_type else str(action.function)
        target_ws_name = str(
            target_station_name
            or self.global_state["robot_locations"].get(agent.robot_id, agent.location)
        )

        if action_type in (
            ActionType.CUT,
            ActionType.COOK,
            ActionType.BOIL,
            ActionType.WASH,
            ActionType.ASSEMBLE,
        ):
            if target_ws_name:
                self.workstation_locks[target_ws_name] = {
                    "locked_by": agent.robot_id,
                    "until_timestep": timestep + duration,
                    "action": func_name,
                }
                print(
                    f"[Environment] 🔒 Workstation '{target_ws_name}' LOCKED by {agent.robot_id} until t={timestep + duration}s."
                )

            tool_item = self._find_required_tool_for_action(
                action_type, target_ws_name, agent
            )
            if tool_item:
                self.tool_locks[tool_item.id] = {
                    "locked_by": agent.robot_id,
                    "tool_name": tool_item.name,
                    "until_timestep": timestep + duration,
                    "action": func_name,
                }
                print(
                    f"[Environment] 🔒 Tool '{tool_item.name}' ({tool_item.id}) LOCKED by {agent.robot_id} until t={timestep + duration}s."
                )

        agent.busy_until = timestep + duration
        agent.current_action = {
            "function": func_name,
            "duration": duration,
            "target_item": action.target_item,
            "station": target_ws_name,
            "start_time": timestep,
            "finish_time": timestep + duration,
            "content": action.content,
        }
        print(f"[Environment] {agent.robot_id} STARTED {func_name} (takes {duration}s)")

    def _reject_physical_action(
        self,
        agent: Any,
        action: Message,
        action_type: Optional[ActionType],
        error_msg: str,
        timestep: int,
    ):
        """Records a rejected physical action and sends an error message back to the offending agent."""
        func_name = action_type.value if action_type else str(action.function)
        print(
            f"[Environment] Action {func_name} by {agent.robot_id} REJECTED: {error_msg}"
        )
        agent.last_action_str = f"FAILED:{func_name}"
        if self.failure_manager:
            self.failure_manager.observe_action_rejection(
                agent.robot_id, error_msg, timestep
            )
        self.failed_actions.append(
            {
                "timestep": timestep,
                "robot_id": agent.robot_id,
                "function": func_name,
                "error": error_msg,
            }
        )
        self.route_message(
            Message(
                sender="System",
                recipient=agent.robot_id,
                timestep=timestep,
                function=ActionType.ERROR,
                intent=CommunicationIntent.FAILURE_EXPLANATION,
                content=f"Action {func_name} rejected: {error_msg}",
                previous_action=agent.last_action_str,
            )
        )

    def _execute_new_actions(self, actions: List[Tuple[Any, Message]], timestep: int):
        """
        Executes single-turn decisions for all ready agents, strictly enforcing the
        'Observe -> reason -> communicate or act -> receive feedback -> update plan' paradigm.
        In each tick, an agent either routes a communication message (Phase 1) OR executes
        a physical action (Phase 2), validating preconditions and handling execution slip failures.
        """
        for agent, action in actions:
            action_type = ActionType.from_string(action.function)

            # 1. Communication turn (communicate OR act)
            if action_type in CONFIG.communication_actions:
                self.route_message(action)
                agent.last_action_str = f"MESSAGE->{action.recipient}[{action.intent.value if isinstance(action.intent, CommunicationIntent) else action.function}]"
                continue

            # 2. Physical action turn: validate & check for injected execution slips
            is_valid, error_msg, target_station_name = self.validator.validate(
                agent, action, self.global_state, env=self
            )
            if is_valid and self.failure_manager:
                slip_fail, slip_err = self.failure_manager.intercept_action_execution(
                    self, agent, action, action_type, timestep
                )
                if slip_fail:
                    is_valid = False
                    error_msg = slip_err

            if is_valid:
                self._start_valid_physical_action(
                    agent, action, action_type, target_station_name, timestep
                )
            else:
                self._reject_physical_action(
                    agent, action, action_type, error_msg, timestep
                )

    def timestep(self, timestep: int):
        """Executes one simulation second (t -> t + 1s)."""
        self.current_time = timestep
        if not self._task_broadcasted:
            self.broadcast_task_at_start()

        print(
            f"\n==================== Timestep {timestep}s (Condition: {self.condition_config.name}) ===================="
        )

        # 0. Expire locks & run FailureManager pre-action triggers
        self._expire_locks(timestep)
        if self.failure_manager:
            self.failure_manager.step_pre_actions(self, timestep)

        # 1. Finalize completed actions
        self._process_completed_actions(timestep)

        # 2. Render workstation status
        self._render_timestep_status(timestep)

        # 3. Update local agent observations
        self._update_agent_observations(timestep)

        # 4. Query idle agents & execute single-turn actions (communicate OR act)
        actions = self._collect_agent_decisions(timestep)
        self._execute_new_actions(actions, timestep)

        # 5. Evaluate post-failure recovery, snapshot state, and notify passive observer
        if self.failure_manager:
            self.failure_manager.evaluate_recovery(self, timestep)
        self.snapshot_state(timestep)
        if self.observer:
            if not self._observer_initialized:
                self.observer.observe_initialization(self)
                self._observer_initialized = True
            self.observer.observe_timestep(self, timestep)

    def compute_metrics(self) -> Tuple[TaskPerformanceMetrics, CommunicationMetrics]:
        """Computes objective Task Performance and Communication metrics for this run."""
        return (
            MetricsEngine.compute_task_metrics(self),
            MetricsEngine.compute_communication_metrics(self),
        )

    def evaluate_and_export_trace(
        self,
        judge: Optional[LLMJudge] = None,
        filepath: Optional[str] = None,
    ) -> Tuple[TaskPerformanceMetrics, CommunicationMetrics, LLMJudgeReport, str]:
        """
        Finalizes the PassiveObserver execution trace, runs the Independent LLM Judge,
        attaches LLM-judge scores to CommunicationMetrics, and exports the trace to
        `runs/<run_id>.json`.
        """
        t_metrics, c_metrics = self.compute_metrics()
        if not self.observer:
            self.observer = PassiveObserver()
        if not self._observer_initialized:
            self.observer.observe_initialization(self)
            self._observer_initialized = True

        self.observer.finalize_observation(
            self, task_metrics=t_metrics, comm_metrics=c_metrics
        )

        active_judge = judge or LLMJudge(
            acting_agent_model=self.observer.acting_agent_model
        )
        j_report = active_judge.evaluate_trace(self.observer)

        # Sync LLM-judge scores back into CommunicationMetrics & observer trace
        c_metrics.llm_judge_scores = j_report.get_scores_dict()
        c_metrics.llm_judge_overall_score = j_report.overall_average_score
        self.observer.objective_metrics["communication"] = c_metrics.to_dict()

        saved_path = self.observer.export_trace(filepath=filepath)
        return t_metrics, c_metrics, j_report, saved_path
