"""
Task Dependency Graph (DAG) and benchmark suite (`tasks.py`).

Defines recipes (`IngredientRequirement`, `Recipe`), DAG subtask milestones (`TaskNode`, `Task`),
ground-truth task verification (`TaskVerifier`), and the 6-task benchmark suite (`TaskBenchmarkSuite`).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import re

from models import ActionType, Item


# ---------------------------------------------------------
# Generalized Recipe & State Transition System
# ---------------------------------------------------------
@dataclass
class IngredientRequirement:
    """Requirement for a specific ingredient name, state, and quantity in a recipe."""
    name: str                       # e.g., "tomato", "lettuce", "patty"
    state: str = "raw"              # required state: "cut", "cooked", "boiled", "washed", "raw"
    count: int = 1                  # number of such items required

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ingredient requirement into a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "state": self.state,
            "count": self.count,
        }


@dataclass
class Recipe:
    """Defines recipe assembly requirements, container, and output dish."""
    name: str                                                           # Recipe canonical identifier
    output_dish: str                                                    # Name of produced dish
    required_ingredients: List[IngredientRequirement] = field(default_factory=list)
    container: Optional[str] = None                                     # e.g., "bowl", "plate", "cup"
    target_station: str = "Food assembly table"
    description: str = ""

    def matches_target(self, query: str) -> bool:
        """Checks whether a query string matches this recipe's canonical or output dish name."""
        if not query:
            return False
        q = query.strip().lower()
        return q in (self.name.lower(), self.output_dish.lower())

    def check_satisfaction(
        self,
        available_ingredients: List[Item],
        available_containers: List[Item]
    ) -> Tuple[bool, List[Item], Optional[Item], str]:
        """
        Checks if available ingredients and containers satisfy all recipe requirements.
        Returns `(is_satisfied, matched_ingredients, matched_container, missing_reason)`.
        """
        matched_container = None
        if self.container:
            for item in available_containers:
                if item.name.lower() == self.container.lower() and item.is_tool:
                    matched_container = item
                    break
            if not matched_container:
                return False, [], None, f"Missing required container '{self.container}'"

        # Match distinct non-tool items for each ingredient requirement
        matched_ingredients: List[Item] = []
        used_ids: Set[str] = set()
        for req in self.required_ingredients:
            found_count = 0
            for item in available_ingredients:
                if item.id in used_ids:
                    continue
                if item.name.lower() == req.name.lower() and item.state.lower() == req.state.lower() and not item.is_tool:
                    matched_ingredients.append(item)
                    used_ids.add(item.id)
                    found_count += 1
                    if found_count >= req.count:
                        break
            if found_count < req.count:
                return False, [], None, f"Missing {req.count - found_count}x '{req.name}' in state '{req.state}' (found {found_count})"

        return True, matched_ingredients, matched_container, ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `Recipe` into a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "output_dish": self.output_dish,
            "required_ingredients": [r.to_dict() for r in self.required_ingredients],
            "container": self.container,
            "target_station": self.target_station,
            "description": self.description,
        }


# ---------------------------------------------------------
# Task Dependency Graph
# ---------------------------------------------------------
@dataclass
class TaskNode:
    """Represents a single subtask milestone node in a Task Dependency Graph (DAG)."""
    node_id: str
    description: str
    action_type: ActionType
    target_station: Optional[str] = None
    target_item: Optional[str] = None
    target_items: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `TaskNode` and its prerequisite edges into a dictionary."""
        return {
            "node_id": self.node_id,
            "description": self.description,
            "action_type": self.action_type.value if isinstance(self.action_type, ActionType) else str(self.action_type),
            "target_station": self.target_station,
            "target_item": self.target_item,
            "target_items": self.target_items,
            "prerequisites": self.prerequisites,
            "completed": self.completed,
        }


def _item_val(it: Any, prop: str, default: Any = "") -> Any:
    if isinstance(it, dict):
        return it.get(prop, default)
    return getattr(it, prop, default)


def _get_item_name(it: Any) -> str:
    return str(_item_val(it, "name", "")).lower()


def _get_item_loc(it: Any) -> str:
    return str(_item_val(it, "location", ""))


def _get_item_state(it: Any) -> str:
    return str(_item_val(it, "state", "")).lower()


def _get_item_is_tool(it: Any) -> bool:
    return bool(_item_val(it, "is_tool", False))


def extract_node_target_items(node_data: Union[TaskNode, Dict[str, Any]]) -> List[str]:
    """Extracts required ingredient or target item names for a DAG node."""
    if isinstance(node_data, TaskNode):
        if getattr(node_data, "target_items", None):
            return [it.lower() for it in node_data.target_items]
        if node_data.target_item and node_data.target_item != "-":
            return [node_data.target_item.lower()]
        desc = node_data.description.lower()
    else:
        if node_data.get("target_items"):
            return [it.lower() for it in node_data["target_items"]]
        if node_data.get("target_item") and node_data.get("target_item") != "-":
            return [str(node_data["target_item"]).lower()]
        desc = str(node_data.get("description", "")).lower()

    found: List[str] = []
    word_mappings = [
        ("tomato", "tomato"),
        ("lettuce", "lettuce"),
        ("cucumber", "cucumber"),
        ("bread", "bread"),
        ("cheese", "cheese"),
        ("orange", "orange"),
        ("meat_patty", "meat_patty"),
        ("patty", "meat_patty"),
        ("patties", "meat_patty"),
        ("burger_bun", "burger_bun"),
        ("bun", "burger_bun"),
        ("buns", "burger_bun"),
        ("carrot", "carrot"),
        ("potato", "potato"),
    ]
    for pattern, canonical in word_mappings:
        if re.search(r"\b" + re.escape(pattern) + r"\b", desc):
            if canonical not in found:
                found.append(canonical)
    return found


def evaluate_dag_node_completion(
    node_id: str,
    node_data: Union[TaskNode, Dict[str, Any]],
    items: List[Any],
    robot_locations: Optional[Dict[str, str]] = None,
    robot_inventories: Optional[Dict[str, List[Any]]] = None,
    all_nodes: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Evaluates whether a DAG node has physically been completed based on the ground-truth items.
    """
    if isinstance(node_data, TaskNode):
        act_raw = node_data.action_type
        act = act_raw.value if isinstance(act_raw, ActionType) else str(act_raw).upper()
        target_station = node_data.target_station
        target_item = (node_data.target_item or "").lower()
        prereqs = node_data.prerequisites
    else:
        act = str(node_data.get("action_type", "")).upper()
        target_station = node_data.get("target_station")
        target_item = str(node_data.get("target_item") or "").lower()
        if target_item == "-":
            target_item = ""
        prereqs = list(node_data.get("prerequisites", []))

    food_items = [it for it in items if not _get_item_is_tool(it)]

    # 1. Fetching / picking items from storage (ActionType.PICK at Refrigerator or fetch_* nodes)
    if (act in ("PICK", "MOVE_ITEMS") and (target_station or "").lower() == "refrigerator") or node_id.startswith("fetch_"):
        origin_station = target_station or "Refrigerator"
        req_items = extract_node_target_items(node_data)
        if not req_items:
            req_items = [
                _get_item_name(it) for it in food_items
                if _get_item_loc(it).lower() == origin_station.lower()
            ]
            if not req_items:
                return True

        # Check each required ingredient:
        # None of the matching items can still be at origin_station!
        found_any = False
        for req_name in req_items:
            matching = [
                it for it in food_items
                if (req_name in _get_item_name(it) or _get_item_name(it) in req_name)
            ]
            if not matching:
                # May have been consumed into an assembled dish
                assembled_exists = any(
                    _get_item_state(it) in ("assembled", "served")
                    for it in food_items
                )
                if assembled_exists:
                    found_any = True
                    continue
                return False
            found_any = True
            # If ANY matching item is still at origin_station (e.g. cucumber in fridge), NOT completed!
            if any(_get_item_loc(it).lower() == origin_station.lower() for it in matching):
                return False

        return found_any

    # 2. Cutting (ActionType.CUT)
    if act == "CUT":
        req_items = extract_node_target_items(node_data)
        if req_items:
            for req in req_items:
                has_cut = any(
                    (req in _get_item_name(it) or _get_item_name(it) in req)
                    and _get_item_state(it) in ("cut", "assembled", "served")
                    for it in food_items
                )
                if not has_cut:
                    return False
            return True
        if target_item:
            return any(
                (target_item in _get_item_name(it) or _get_item_name(it) in target_item)
                and _get_item_state(it) in ("cut", "assembled", "served")
                for it in food_items
            )
        return any(_get_item_state(it) in ("cut", "assembled", "served") for it in food_items)

    # 3. Cooking (ActionType.COOK)
    if act == "COOK":
        target = target_item or "meat_patty"
        if node_id == "cook_patty_2":
            cooked_cnt = sum(
                1 for it in food_items
                if (target in _get_item_name(it) or _get_item_name(it) in target)
                and _get_item_state(it) in ("cooked", "assembled", "served")
            )
            return cooked_cnt >= 2
        return any(
            (target in _get_item_name(it) or _get_item_name(it) in target)
            and _get_item_state(it) in ("cooked", "assembled", "served")
            for it in food_items
        )

    # 4. Boiling (ActionType.BOIL)
    if act == "BOIL":
        req_items = extract_node_target_items(node_data)
        if req_items:
            for req in req_items:
                has_boiled = any(
                    (req in _get_item_name(it) or _get_item_name(it) in req)
                    and _get_item_state(it) in ("boiled", "assembled", "served")
                    for it in food_items
                )
                if not has_boiled:
                    return False
            return True
        target = target_item
        return any(
            (not target or target in _get_item_name(it) or _get_item_name(it) in target)
            and _get_item_state(it) in ("boiled", "assembled", "served")
            for it in food_items
        )

    # 5. Washing (ActionType.WASH)
    if act == "WASH":
        req_items = extract_node_target_items(node_data)
        if req_items:
            for req in req_items:
                has_washed = any(
                    (req in _get_item_name(it) or _get_item_name(it) in req)
                    and _get_item_state(it) in ("washed", "cut", "assembled", "served")
                    for it in food_items
                )
                if not has_washed:
                    return False
            return True
        target = target_item
        return any(
            (not target or target in _get_item_name(it) or _get_item_name(it) in target)
            and _get_item_state(it) in ("washed", "cut", "assembled", "served")
            for it in food_items
        )

    # 6. Assembling (ActionType.ASSEMBLE)
    if act == "ASSEMBLE":
        target = target_item
        if node_id == "assemble_burger_2":
            ass_cnt = sum(
                1 for it in food_items
                if (target in _get_item_name(it) or _get_item_name(it) in target)
                and _get_item_state(it) in ("assembled", "served")
            )
            return ass_cnt >= 2
        return any(
            (not target or target in _get_item_name(it) or _get_item_name(it) in target)
            and _get_item_state(it) in ("assembled", "served")
            for it in food_items
        )

    # 7. Dropping / Transport / Serving (ActionType.DROP)
    if act == "DROP":
        target_ws = (target_station or "").lower()
        if target_ws == "serving table" or node_id.startswith("serve_"):
            target = target_item
            if target:
                return any(
                    (target in _get_item_name(it) or _get_item_name(it) in target)
                    and _get_item_loc(it).lower() == "serving table"
                    and _get_item_state(it) in ("assembled", "served", "ready")
                    for it in food_items
                )
            else:
                served = [
                    it for it in food_items
                    if _get_item_loc(it).lower() == "serving table"
                    and _get_item_state(it) in ("assembled", "served", "ready")
                ]
                return len(served) > 0

        # Intermediate transport (e.g. transport_to_assembly, transport_veggies, transport_patty, transport_soup_to_stove)
        req_items = []
        if target_item:
            req_items = [target_item]
        elif all_nodes and prereqs:
            for p_id in prereqs:
                p_data = all_nodes.get(p_id)
                if p_data:
                    req_items.extend(extract_node_target_items(p_data))
        if not req_items:
            req_items = extract_node_target_items(node_data)

        if req_items:
            for req in req_items:
                matching = [
                    it for it in food_items
                    if (req in _get_item_name(it) or _get_item_name(it) in req)
                ]
                if not matching:
                    assembled_exists = any(
                        _get_item_state(it) in ("assembled", "served")
                        for it in food_items
                    )
                    if assembled_exists:
                        continue
                    return False

                item_reached = False
                for it in matching:
                    loc = _get_item_loc(it)
                    if loc.lower() == target_ws:
                        item_reached = True
                        break
                    if robot_locations and loc in robot_locations and robot_locations[loc].lower() == target_ws:
                        item_reached = True
                        break
                    if _get_item_state(it) in ("assembled", "served"):
                        item_reached = True
                        break
                if not item_reached:
                    return False

            # Check that the destination station also has the required tool for downstream operations
            has_assembled_dish = any(
                _get_item_state(it) in ("assembled", "served")
                and _get_item_loc(it).lower() in ("food assembly table", "serving table")
                for it in food_items
            )
            if not has_assembled_dish:
                tool_items = [it for it in items if _get_item_is_tool(it)]
                station_tools = [
                    _get_item_name(it) for it in tool_items
                    if _get_item_loc(it).lower() == target_ws
                    or (robot_locations and _get_item_loc(it) in robot_locations and robot_locations[_get_item_loc(it)].lower() == target_ws)
                ]
                if target_ws == "food assembly table":
                    if not any(t in ("bowl", "plate", "cup") for t in station_tools):
                        return False
                elif target_ws == "stove":
                    if not any(t in ("pan", "pot") for t in station_tools):
                        return False
                elif target_ws == "cutting table":
                    if not any(t == "knife" for t in station_tools):
                        return False

            return True

        return any(_get_item_loc(it).lower() == target_ws for it in food_items)

    return False


def evaluate_dag_node_status(
    node_id: str,
    node_data: Union[TaskNode, Dict[str, Any]],
    all_nodes: Dict[str, Any],
    items: List[Any],
    robot_locations: Optional[Dict[str, str]] = None,
    robot_inventories: Optional[Dict[str, List[Any]]] = None,
    completed_cache: Optional[Dict[str, bool]] = None,
) -> Tuple[str, List[str]]:
    """
    Evaluates whether a DAG node is "completed", "ready" (unblocked), or "blocked"
    at the given physical state, along with any missing prerequisite IDs or reasons.
    """
    if completed_cache is None:
        completed_cache = {}

    def is_comp(n_id: str, n_val: Any) -> bool:
        if n_id not in completed_cache:
            completed_cache[n_id] = evaluate_dag_node_completion(
                n_id,
                n_val,
                items,
                robot_locations=robot_locations,
                robot_inventories=robot_inventories,
                all_nodes=all_nodes,
            )
        return completed_cache[n_id]

    # 1. Check if this node itself is completed
    if is_comp(node_id, node_data):
        return "completed", []

    # 2. Check prerequisite nodes
    if isinstance(node_data, TaskNode):
        prereqs = list(node_data.prerequisites)
        act = node_data.action_type.value if isinstance(node_data.action_type, ActionType) else str(node_data.action_type).upper()
        target_item = (node_data.target_item or "").lower()
        target_st = (node_data.target_station or "").lower()
    else:
        prereqs = list(node_data.get("prerequisites", []))
        act = str(node_data.get("action_type", "")).upper()
        target_item = str(node_data.get("target_item") or "").lower()
        if target_item == "-":
            target_item = ""
        target_st = str(node_data.get("target_station") or "").lower()

    missing = []
    for p_id in prereqs:
        p_node = all_nodes.get(p_id)
        if p_node is not None:
            if not is_comp(p_id, p_node):
                missing.append(p_id)
        else:
            missing.append(p_id)

    if missing:
        return "blocked", missing

    # 3. Physical readiness check:
    # If this is a station transformation node (CUT, COOK, BOIL, WASH),
    # check that its target item(s) are NOT still in Refrigerator (storage)
    if act in ("CUT", "COOK", "BOIL", "WASH"):
        chk_items = extract_node_target_items(node_data)
        if not chk_items and target_item:
            chk_items = [target_item]
        if chk_items:
            food_items = [it for it in items if not _get_item_is_tool(it)]
            for chk in chk_items:
                matching = [
                    it for it in food_items
                    if (chk in _get_item_name(it) or _get_item_name(it) in chk)
                ]
                if matching and all(_get_item_loc(it).lower() == "refrigerator" and _get_item_state(it) == "raw" for it in matching):
                    return "blocked", [f"{chk} still in Refrigerator"]

    # 4. Tool readiness check:
    # If this is a station operation or transport, check that the required tool is available at the workstation
    tool_items = [it for it in items if _get_item_is_tool(it)]
    station_tools = [
        _get_item_name(it) for it in tool_items
        if _get_item_loc(it).lower() == target_st
        or (robot_locations and _get_item_loc(it) in robot_locations and robot_locations[_get_item_loc(it)].lower() == target_st)
    ]
    if act == "CUT" and target_st == "cutting table":
        if not any("knife" in t for t in station_tools):
            return "blocked", ["knife not at Cutting table"]
    elif act == "COOK" and target_st == "stove":
        if not any(t in ("pan", "pot") for t in station_tools):
            return "blocked", ["cookware not at Stove"]
    elif act == "BOIL" and target_st == "stove":
        if not any("pot" in t for t in station_tools):
            return "blocked", ["pot not at Stove"]
    elif act == "ASSEMBLE" and target_st == "food assembly table":
        if not any(t in ("bowl", "plate", "cup") for t in station_tools):
            return "blocked", ["container tool not at Food assembly table"]
    elif act == "DROP" and target_st == "food assembly table":
        if not any(t in ("bowl", "plate", "cup") for t in station_tools):
            return "blocked", ["container tool not at Food assembly table"]
    elif act == "DROP" and target_st == "cutting table":
        if not any("knife" in t for t in station_tools):
            return "blocked", ["knife not at Cutting table"]
    elif act == "DROP" and target_st == "stove":
        if not any(t in ("pan", "pot") for t in station_tools):
            return "blocked", ["cookware not at Stove"]

    return "ready", []


@dataclass
class Task:
    """Encapsulates a complete benchmark task specification, initial items, recipes, and DAG."""
    task_id: str
    name: str
    difficulty: str                         # "Easy", "Medium", "Hard"
    description: str
    target_dishes: List[str]                # e.g., ["salad"], ["sandwich", "fruit_drink"]
    target_station: str = "Serving table"   # Target workstation for final delivery
    required_tools: List[str] = field(default_factory=list)
    initial_items: List[Item] = field(default_factory=list)
    recipes: List[Recipe] = field(default_factory=list)
    dependency_graph: Dict[str, TaskNode] = field(default_factory=dict)
    min_capabilities: Dict[str, Any] = field(default_factory=dict)
    parallel_opportunities: List[str] = field(default_factory=list)

    def is_node_ready(self, node_id: str) -> bool:
        """Returns True if `node_id` is not completed and all its prerequisites are completed."""
        node = self.dependency_graph.get(node_id)
        if not node or node.completed:
            return False
        return all(
            self.dependency_graph[p].completed
            for p in node.prerequisites
            if p in self.dependency_graph
        )

    def sync_dag(
        self,
        items: List[Any],
        robot_locations: Optional[Dict[str, str]] = None,
        robot_inventories: Optional[Dict[str, List[Any]]] = None,
    ):
        """Updates DAG node completion status from ground-truth physical item states."""
        for node_id, node in self.dependency_graph.items():
            if node.completed:
                continue
            if evaluate_dag_node_completion(
                node_id,
                node,
                items,
                robot_locations=robot_locations,
                robot_inventories=robot_inventories,
                all_nodes=self.dependency_graph,
            ):
                node.completed = True

    def get_progress(self) -> Dict[str, Any]:
        """Returns current DAG node counts and lists of pending, ready, and completed nodes."""
        total = len(self.dependency_graph)
        completed = sum(1 for n in self.dependency_graph.values() if n.completed)
        return {
            "total_nodes": total,
            "completed_nodes": completed,
            # ###Legacy metrics that were not used in the final analysis###
            "percent": round((completed / total * 100.0), 1) if total > 0 else 0.0,
            "pending_nodes": [n.node_id for n in self.dependency_graph.values() if not n.completed],
            "ready_nodes": [n.node_id for n in self.dependency_graph.values() if self.is_node_ready(n.node_id)],
            "completed_list": [n.node_id for n in self.dependency_graph.values() if n.completed],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `Task` specification and DAG progress to a dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "target_dishes": self.target_dishes,
            "target_station": self.target_station,
            "required_tools": self.required_tools,
            "min_capabilities": self.min_capabilities,
            "parallel_opportunities": self.parallel_opportunities,
            "recipes": [r.to_dict() for r in self.recipes],
            "dependency_graph": {k: n.to_dict() for k, n in self.dependency_graph.items()},
            "progress": self.get_progress(),
        }


# ---------------------------------------------------------
# Ground-Truth Task Completion Verifier
# ---------------------------------------------------------
@dataclass
class TaskVerificationResult:
    """Structured result returned by `TaskVerifier.verify` after inspecting simulator state."""
    is_complete: bool
    target_dishes_found: List[str]
    missing_dishes: List[str]
    target_station: str
    completed_milestones: List[str]
    missing_milestones: List[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the verification outcome to a JSON-compatible dictionary."""
        return {
            "is_complete": self.is_complete,
            "target_dishes_found": self.target_dishes_found,
            "missing_dishes": self.missing_dishes,
            "target_station": self.target_station,
            "completed_milestones": self.completed_milestones,
            "missing_milestones": self.missing_milestones,
            "details": self.details,
        }


class TaskVerifier:
    """Deterministic ground-truth task completion verifier."""

    @staticmethod
    def verify(env: Any, task: Optional[Task] = None) -> TaskVerificationResult:
        """Checks whether all required target dishes are assembled/served at the target station."""
        active_task = task or getattr(env, "current_task", None)
        if not active_task:
            return TaskVerificationResult(
                is_complete=False,
                target_dishes_found=[],
                missing_dishes=[],
                target_station="Unknown",
                completed_milestones=[],
                missing_milestones=[],
                details={"error": "No active task loaded in environment."}
            )

        target_station = active_task.target_station
        station_items = env.get_items_at_station(target_station)

        found_dishes: List[str] = []
        missing_dishes: List[str] = []
        used_item_ids: Set[str] = set()

        # Match each required dish to a distinct assembled/served item at target_station
        for req_dish in active_task.target_dishes:
            req_clean = req_dish.lower().split("_")[0]
            matched_item = None
            for item in station_items:
                if item.id in used_item_ids:
                    continue
                item_clean = item.name.lower().split("_")[0]
                if (req_clean == item_clean or req_clean in item.id.lower()) and item.state in ("assembled", "served", "ready"):
                    matched_item = item
                    break

            if matched_item:
                found_dishes.append(matched_item.name)
                used_item_ids.add(matched_item.id)
            else:
                missing_dishes.append(req_dish)

        progress = active_task.get_progress()
        is_complete = (len(missing_dishes) == 0 and len(found_dishes) >= len(active_task.target_dishes))

        return TaskVerificationResult(
            is_complete=is_complete,
            target_dishes_found=found_dishes,
            missing_dishes=missing_dishes,
            target_station=target_station,
            completed_milestones=progress.get("completed_list", []),
            missing_milestones=progress.get("pending_nodes", []),
            details={
                "task_name": active_task.name,
                "difficulty": active_task.difficulty,
                "progress_percent": progress.get("percent", 0.0),
                "items_at_target_station": [f"{it.name} ({it.state})" for it in station_items]
            }
        )


# ---------------------------------------------------------
# Task Benchmark Suite
# ---------------------------------------------------------
class TaskBenchmarkSuite:
    """Benchmark factory suite providing the 6 standard multi-robot kitchen tasks."""

    @staticmethod
    def task_1_prepare_salad() -> Task:
        """
        Constructs Task 1 (Easy): Prepare a salad.
        Tests basic task allocation: fetching tomato, lettuce, and cucumber from the
        Refrigerator, cutting them with the knife at the Cutting table, assembling them
        in a bowl at the Food assembly table, and delivering to the Serving table.
        """
        from config import DEFAULT_RECIPES
        recipe = DEFAULT_RECIPES["salad"]
        initial_items = [
            Item(id="knife_1", name="knife", location="Cutting table", is_tool=True, state="ready"),
            Item(id="bowl_1", name="bowl", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="tomato_1", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_1", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="cucumber_1", name="cucumber", location="Refrigerator", is_tool=False, state="raw"),
        ]
        dag = {
            "fetch_veggies": TaskNode("fetch_veggies", "Fetch tomato, lettuce, cucumber from Refrigerator", ActionType.PICK, "Refrigerator", target_items=["tomato", "lettuce", "cucumber"]),
            "cut_tomato": TaskNode("cut_tomato", "Cut tomato with knife at Cutting table", ActionType.CUT, "Cutting table", "tomato", prerequisites=["fetch_veggies"]),
            "cut_lettuce": TaskNode("cut_lettuce", "Cut lettuce with knife at Cutting table", ActionType.CUT, "Cutting table", "lettuce", prerequisites=["fetch_veggies"]),
            "cut_cucumber": TaskNode("cut_cucumber", "Cut cucumber with knife at Cutting table", ActionType.CUT, "Cutting table", "cucumber", prerequisites=["fetch_veggies"]),
            "transport_to_assembly": TaskNode("transport_to_assembly", "Transport cut vegetables to Food assembly table", ActionType.DROP, "Food assembly table", prerequisites=["cut_tomato", "cut_lettuce", "cut_cucumber"]),
            "assemble_salad": TaskNode("assemble_salad", "Assemble salad in bowl at Food assembly table", ActionType.ASSEMBLE, "Food assembly table", "salad", prerequisites=["transport_to_assembly"]),
            "serve_salad": TaskNode("serve_salad", "Deliver salad to Serving table", ActionType.DROP, "Serving table", "salad", prerequisites=["assemble_salad"]),
        }
        return Task(
            task_id="task_1",
            name="Prepare a salad",
            difficulty="Easy",
            description="Prepare a fresh salad by fetching vegetables from the refrigerator, cutting them at the cutting table, assembling them into a bowl, and delivering the salad to the serving table.",
            target_dishes=["salad"],
            target_station="Serving table",
            required_tools=["knife", "bowl"],
            initial_items=initial_items,
            recipes=[recipe],
            dependency_graph=dag,
            min_capabilities={"skills": [ActionType.PICK, ActionType.DROP, ActionType.CUT, ActionType.ASSEMBLE], "min_robots": 2},
            parallel_opportunities=["Robots can fetch items and cut vegetables in parallel pipeline."]
        )

    @staticmethod
    def task_2_prepare_sandwich_and_drink() -> Task:
        """
        Constructs Task 2 (Easy): Prepare a sandwich and drink.
        Tests parallel branch execution across two independent recipes (`sandwich` and `fruit_drink`).
        """
        from config import DEFAULT_RECIPES
        r_sandwich = DEFAULT_RECIPES["sandwich"]
        r_drink = DEFAULT_RECIPES["fruit_drink"]
        initial_items = [
            Item(id="knife_1", name="knife", location="Cutting table", is_tool=True, state="ready"),
            Item(id="plate_1", name="plate", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="cup_1", name="cup", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="bread_1", name="bread", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="cheese_1", name="cheese", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_1", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="orange_1", name="orange", location="Refrigerator", is_tool=False, state="raw"),
        ]
        dag = {
            "fetch_sandwich_ingredients": TaskNode("fetch_sandwich_ingredients", "Fetch bread, cheese, tomato from Refrigerator", ActionType.PICK, "Refrigerator", target_items=["bread", "cheese", "tomato"]),
            "cut_cheese": TaskNode("cut_cheese", "Cut cheese at Cutting table", ActionType.CUT, "Cutting table", "cheese", prerequisites=["fetch_sandwich_ingredients"]),
            "cut_tomato": TaskNode("cut_tomato", "Cut tomato at Cutting table", ActionType.CUT, "Cutting table", "tomato", prerequisites=["fetch_sandwich_ingredients"]),
            "assemble_sandwich": TaskNode("assemble_sandwich", "Assemble sandwich on plate", ActionType.ASSEMBLE, "Food assembly table", "sandwich", prerequisites=["cut_cheese", "cut_tomato"]),
            "serve_sandwich": TaskNode("serve_sandwich", "Deliver sandwich to Serving table", ActionType.DROP, "Serving table", "sandwich", prerequisites=["assemble_sandwich"]),
            "fetch_orange": TaskNode("fetch_orange", "Fetch orange from Refrigerator", ActionType.PICK, "Refrigerator", target_items=["orange"]),
            "wash_orange": TaskNode("wash_orange", "Wash orange at Sink", ActionType.WASH, "Sink", "orange", prerequisites=["fetch_orange"]),
            "cut_orange": TaskNode("cut_orange", "Cut orange at Cutting table", ActionType.CUT, "Cutting table", "orange", prerequisites=["wash_orange"]),
            "assemble_drink": TaskNode("assemble_drink", "Assemble drink in cup at Food assembly table", ActionType.ASSEMBLE, "Food assembly table", "fruit_drink", prerequisites=["cut_orange"]),
            "serve_drink": TaskNode("serve_drink", "Deliver drink to Serving table", ActionType.DROP, "Serving table", "fruit_drink", prerequisites=["assemble_drink"]),
        }
        return Task(
            task_id="task_2",
            name="Prepare a sandwich and drink",
            difficulty="Easy",
            description="Prepare a sandwich and a fruit drink simultaneously, exploiting two independent parallel branches.",
            target_dishes=["sandwich", "fruit_drink"],
            target_station="Serving table",
            required_tools=["knife", "plate", "cup"],
            initial_items=initial_items,
            recipes=[r_sandwich, r_drink],
            dependency_graph=dag,
            min_capabilities={"skills": [ActionType.PICK, ActionType.DROP, ActionType.CUT, ActionType.WASH, ActionType.ASSEMBLE], "min_robots": 2},
            parallel_opportunities=["One robot prepares sandwich while second robot prepares drink at Sink and Assembly table."]
        )

    @staticmethod
    def task_3_prepare_hamburgers() -> Task:
        """
        Constructs Task 3 (Medium): Prepare hamburgers.
        Tests multi-role sequential dependencies (cooking meat patty on Stove while cutting
        vegetables at Cutting table, then converging at Food assembly table).
        """
        from config import DEFAULT_RECIPES
        r_burger = DEFAULT_RECIPES["hamburger"]
        initial_items = [
            Item(id="pan_1", name="pan", location="Stove", is_tool=True, state="ready"),
            Item(id="knife_1", name="knife", location="Cutting table", is_tool=True, state="ready"),
            Item(id="plate_1", name="plate", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="patty_1", name="meat_patty", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="burger_bun_1", name="burger_bun", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_1", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_1", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
        ]
        dag = {
            "fetch_patty": TaskNode("fetch_patty", "Fetch patty from Refrigerator to Stove", ActionType.PICK, "Refrigerator", "meat_patty", target_items=["meat_patty"]),
            "cook_patty": TaskNode("cook_patty", "Cook meat patty on Stove with pan", ActionType.COOK, "Stove", "meat_patty", prerequisites=["fetch_patty"]),
            "transport_patty": TaskNode("transport_patty", "Move cooked patty to Food assembly table", ActionType.DROP, "Food assembly table", "meat_patty", prerequisites=["cook_patty"]),
            "fetch_veggies": TaskNode("fetch_veggies", "Fetch lettuce, tomato from Refrigerator", ActionType.PICK, "Refrigerator", target_items=["lettuce", "tomato"]),
            "cut_lettuce": TaskNode("cut_lettuce", "Cut lettuce at Cutting table", ActionType.CUT, "Cutting table", "lettuce", prerequisites=["fetch_veggies"]),
            "cut_tomato": TaskNode("cut_tomato", "Cut tomato at Cutting table", ActionType.CUT, "Cutting table", "tomato", prerequisites=["fetch_veggies"]),
            "transport_veggies": TaskNode("transport_veggies", "Transport cut veggies to Food assembly table", ActionType.DROP, "Food assembly table", prerequisites=["cut_lettuce", "cut_tomato"]),
            "fetch_bun": TaskNode("fetch_bun", "Fetch burger bun to Food assembly table", ActionType.PICK, "Refrigerator", "burger_bun", target_items=["burger_bun"]),
            "assemble_burger": TaskNode("assemble_burger", "Assemble hamburger on plate with cooked patty, bun, and cut veggies", ActionType.ASSEMBLE, "Food assembly table", "hamburger", prerequisites=["transport_patty", "transport_veggies", "fetch_bun"]),
            "serve_burger": TaskNode("serve_burger", "Deliver assembled hamburger to Serving table", ActionType.DROP, "Serving table", "hamburger", prerequisites=["assemble_burger"]),
        }
        return Task(
            task_id="task_3",
            name="Prepare hamburgers",
            difficulty="Medium",
            description="Cook meat patty on the stove while cutting vegetables, then assemble into a hamburger on a plate and serve.",
            target_dishes=["hamburger"],
            target_station="Serving table",
            required_tools=["pan", "knife", "plate"],
            initial_items=initial_items,
            recipes=[r_burger],
            dependency_graph=dag,
            min_capabilities={"skills": [ActionType.PICK, ActionType.DROP, ActionType.COOK, ActionType.CUT, ActionType.ASSEMBLE], "min_robots": 2},
            parallel_opportunities=["Cooking patty on Stove can run in parallel with cutting vegetables on Cutting table."]
        )

    @staticmethod
    def task_4_prepare_soup_and_salad() -> Task:
        """
        Constructs Task 4 (Medium): Prepare soup and salad.
        Tests shared tool and station contention over the single `knife_1` and `Cutting table`.
        """
        from config import DEFAULT_RECIPES
        r_soup = DEFAULT_RECIPES["vegetable_soup"]
        r_salad = DEFAULT_RECIPES["salad"]
        initial_items = [
            Item(id="knife_1", name="knife", location="Cutting table", is_tool=True, state="ready"),
            Item(id="pot_1", name="pot", location="Stove", is_tool=True, state="ready"),
            Item(id="bowl_1", name="bowl", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="bowl_2", name="bowl", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="carrot_1", name="carrot", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="potato_1", name="potato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_1", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_1", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="cucumber_1", name="cucumber", location="Refrigerator", is_tool=False, state="raw"),
        ]
        dag = {
            "fetch_soup_veggies": TaskNode("fetch_soup_veggies", "Fetch carrot and potato to Cutting table", ActionType.PICK, "Refrigerator", target_items=["carrot", "potato"]),
            "cut_carrot": TaskNode("cut_carrot", "Cut carrot at Cutting table (requires knife_1)", ActionType.CUT, "Cutting table", "carrot", prerequisites=["fetch_soup_veggies"]),
            "cut_potato": TaskNode("cut_potato", "Cut potato at Cutting table (requires knife_1)", ActionType.CUT, "Cutting table", "potato", prerequisites=["fetch_soup_veggies"]),
            "transport_soup_to_stove": TaskNode("transport_soup_to_stove", "Move cut carrot and potato to Stove", ActionType.DROP, "Stove", prerequisites=["cut_carrot", "cut_potato"]),
            "boil_soup": TaskNode("boil_soup", "Boil vegetables in pot at Stove", ActionType.BOIL, "Stove", target_items=["carrot", "potato"], prerequisites=["transport_soup_to_stove"]),
            "assemble_soup": TaskNode("assemble_soup", "Assemble vegetable soup in bowl", ActionType.ASSEMBLE, "Food assembly table", "vegetable_soup", prerequisites=["boil_soup"]),
            "serve_soup": TaskNode("serve_soup", "Deliver soup to Serving table", ActionType.DROP, "Serving table", "vegetable_soup", prerequisites=["assemble_soup"]),
            "fetch_salad_veggies": TaskNode("fetch_salad_veggies", "Fetch lettuce, tomato, cucumber to Cutting table", ActionType.PICK, "Refrigerator", target_items=["lettuce", "tomato", "cucumber"]),
            "cut_salad_veggies": TaskNode("cut_salad_veggies", "Cut salad veggies at Cutting table (contends for knife_1)", ActionType.CUT, "Cutting table", target_items=["lettuce", "tomato", "cucumber"], prerequisites=["fetch_salad_veggies"]),
            "assemble_salad": TaskNode("assemble_salad", "Assemble salad in bowl", ActionType.ASSEMBLE, "Food assembly table", "salad", prerequisites=["cut_salad_veggies"]),
            "serve_salad": TaskNode("serve_salad", "Deliver salad to Serving table", ActionType.DROP, "Serving table", "salad", prerequisites=["assemble_salad"]),
        }
        return Task(
            task_id="task_4",
            name="Prepare soup and salad",
            difficulty="Medium",
            description="Prepare vegetable soup and garden salad with high contention over the single knife and cutting station.",
            target_dishes=["vegetable_soup", "salad"],
            target_station="Serving table",
            required_tools=["knife", "pot", "bowl"],
            initial_items=initial_items,
            recipes=[r_soup, r_salad],
            dependency_graph=dag,
            min_capabilities={"skills": [ActionType.PICK, ActionType.DROP, ActionType.CUT, ActionType.BOIL, ActionType.ASSEMBLE], "min_robots": 2},
            parallel_opportunities=["Soup boiling at Stove can run concurrently with salad assembly once cutting finishes."]
        )

    @staticmethod
    def task_5_prepare_two_hamburgers_and_salad() -> Task:
        """
        Constructs Task 5 (Hard): Prepare two hamburgers and a salad.
        Tests intense resource competition across the Stove (`pan_1`), Cutting table (`knife_1`),
        and Food assembly table (`plate_1`, `plate_2`, `bowl_1`).
        """
        from config import DEFAULT_RECIPES
        r_burger = DEFAULT_RECIPES["hamburger"]
        r_salad = DEFAULT_RECIPES["salad"]
        initial_items = [
            Item(id="pan_1", name="pan", location="Stove", is_tool=True, state="ready"),
            Item(id="knife_1", name="knife", location="Cutting table", is_tool=True, state="ready"),
            Item(id="plate_1", name="plate", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="plate_2", name="plate", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="bowl_1", name="bowl", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="patty_1", name="meat_patty", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="patty_2", name="meat_patty", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="burger_bun_1", name="burger_bun", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="burger_bun_2", name="burger_bun", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_1", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_2", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_3", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_1", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_2", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_3", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="cucumber_1", name="cucumber", location="Refrigerator", is_tool=False, state="raw"),
        ]
        dag = {
            "fetch_all": TaskNode("fetch_all", "Fetch patties, buns, and vegetables from Refrigerator", ActionType.PICK, "Refrigerator", target_items=["meat_patty", "burger_bun", "lettuce", "tomato", "cucumber"]),
            "cook_patty_1": TaskNode("cook_patty_1", "Cook first patty on Stove (locks pan_1)", ActionType.COOK, "Stove", "meat_patty", prerequisites=["fetch_all"]),
            "cook_patty_2": TaskNode("cook_patty_2", "Cook second patty on Stove (queued after patty 1)", ActionType.COOK, "Stove", "meat_patty", prerequisites=["cook_patty_1"]),
            "cut_burger_veggies": TaskNode("cut_burger_veggies", "Cut burger lettuce and tomato", ActionType.CUT, "Cutting table", target_items=["lettuce", "tomato"], prerequisites=["fetch_all"]),
            "cut_salad_veggies": TaskNode("cut_salad_veggies", "Cut salad cucumber, lettuce, and tomato", ActionType.CUT, "Cutting table", target_items=["cucumber", "lettuce", "tomato"], prerequisites=["cut_burger_veggies"]),
            "assemble_burger_1": TaskNode("assemble_burger_1", "Assemble first hamburger on plate_1", ActionType.ASSEMBLE, "Food assembly table", "hamburger", prerequisites=["cook_patty_1", "cut_burger_veggies"]),
            "assemble_burger_2": TaskNode("assemble_burger_2", "Assemble second hamburger on plate_2", ActionType.ASSEMBLE, "Food assembly table", "hamburger", prerequisites=["cook_patty_2", "cut_burger_veggies"]),
            "assemble_salad": TaskNode("assemble_salad", "Assemble salad in bowl_1", ActionType.ASSEMBLE, "Food assembly table", "salad", prerequisites=["cut_salad_veggies"]),
            "serve_all": TaskNode("serve_all", "Serve both burgers and salad to Serving table", ActionType.DROP, "Serving table", prerequisites=["assemble_burger_1", "assemble_burger_2", "assemble_salad"]),
        }
        return Task(
            task_id="task_5",
            name="Prepare two hamburgers and a salad",
            difficulty="Hard",
            description="Prepare two cooked hamburgers and a salad under severe resource competition for the stove, cutting table, knife, and assembly table.",
            target_dishes=["hamburger", "hamburger", "salad"],
            target_station="Serving table",
            required_tools=["pan", "knife", "plate", "bowl"],
            initial_items=initial_items,
            recipes=[r_burger, r_salad],
            dependency_graph=dag,
            min_capabilities={"skills": [ActionType.PICK, ActionType.DROP, ActionType.COOK, ActionType.CUT, ActionType.ASSEMBLE], "min_robots": 3},
            parallel_opportunities=["Robots can interleave cutting and stove cooking while coordinating table access."]
        )

    @staticmethod
    def task_6_prepare_complete_meal() -> Task:
        """
        Constructs Task 6 (Hard): Prepare a complete meal.
        Tests full-scale multi-stage planning across all 6 kitchen stations (`Refrigerator`,
        `Sink`, `Cutting table`, `Stove`, `Food assembly table`, `Serving table`) to produce
        a `hamburger`, `vegetable_soup`, and `salad`.
        """
        from config import DEFAULT_RECIPES
        r_burger = DEFAULT_RECIPES["hamburger"]
        r_soup = DEFAULT_RECIPES["vegetable_soup"]
        r_salad = DEFAULT_RECIPES["salad"]
        initial_items = [
            Item(id="pan_1", name="pan", location="Stove", is_tool=True, state="ready"),
            Item(id="pot_1", name="pot", location="Stove", is_tool=True, state="ready"),
            Item(id="knife_1", name="knife", location="Cutting table", is_tool=True, state="ready"),
            Item(id="plate_1", name="plate", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="bowl_1", name="bowl", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="bowl_2", name="bowl", location="Food assembly table", is_tool=True, state="ready"),
            Item(id="patty_1", name="meat_patty", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="burger_bun_1", name="burger_bun", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="carrot_1", name="carrot", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="potato_1", name="potato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_1", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="lettuce_2", name="lettuce", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_1", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="tomato_2", name="tomato", location="Refrigerator", is_tool=False, state="raw"),
            Item(id="cucumber_1", name="cucumber", location="Refrigerator", is_tool=False, state="raw"),
        ]
        dag = {
            "fetch_all_storage": TaskNode("fetch_all_storage", "Fetch all ingredients from Refrigerator", ActionType.PICK, "Refrigerator", target_items=["meat_patty", "burger_bun", "carrot", "potato", "lettuce", "tomato", "cucumber"]),
            "wash_soup_veggies": TaskNode("wash_soup_veggies", "Wash soup vegetables at Sink", ActionType.WASH, "Sink", target_items=["carrot", "potato"], prerequisites=["fetch_all_storage"]),
            "cut_soup_veggies": TaskNode("cut_soup_veggies", "Cut carrot and potato at Cutting table", ActionType.CUT, "Cutting table", target_items=["carrot", "potato"], prerequisites=["wash_soup_veggies"]),
            "boil_soup": TaskNode("boil_soup", "Boil soup in pot on Stove", ActionType.BOIL, "Stove", target_items=["carrot", "potato"], prerequisites=["cut_soup_veggies"]),
            "assemble_soup": TaskNode("assemble_soup", "Assemble soup in bowl_1", ActionType.ASSEMBLE, "Food assembly table", "vegetable_soup", prerequisites=["boil_soup"]),
            "cook_patty": TaskNode("cook_patty", "Cook patty in pan on Stove", ActionType.COOK, "Stove", "meat_patty", prerequisites=["fetch_all_storage"]),
            "wash_salad_veggies": TaskNode("wash_salad_veggies", "Wash salad veggies at Sink", ActionType.WASH, "Sink", target_items=["lettuce", "tomato", "cucumber"], prerequisites=["fetch_all_storage"]),
            "cut_salad_veggies": TaskNode("cut_salad_veggies", "Cut lettuce, tomato, cucumber", ActionType.CUT, "Cutting table", target_items=["lettuce", "tomato", "cucumber"], prerequisites=["wash_salad_veggies", "cut_soup_veggies"]),
            "assemble_burger": TaskNode("assemble_burger", "Assemble burger on plate_1", ActionType.ASSEMBLE, "Food assembly table", "hamburger", prerequisites=["cook_patty", "cut_salad_veggies"]),
            "assemble_salad": TaskNode("assemble_salad", "Assemble salad in bowl_2", ActionType.ASSEMBLE, "Food assembly table", "salad", prerequisites=["cut_salad_veggies"]),
            "serve_complete_meal": TaskNode("serve_complete_meal", "Serve hamburger, soup, and salad to Serving table", ActionType.DROP, "Serving table", prerequisites=["assemble_soup", "assemble_burger", "assemble_salad"]),
        }
        return Task(
            task_id="task_6",
            name="Prepare a complete meal",
            difficulty="Hard",
            description="Prepare a multi-course dinner including hamburger, vegetable soup, and salad utilizing all 6 kitchen stations and full parallel robot execution.",
            target_dishes=["hamburger", "vegetable_soup", "salad"],
            target_station="Serving table",
            required_tools=["pan", "pot", "knife", "plate", "bowl"],
            initial_items=initial_items,
            recipes=[r_burger, r_soup, r_salad],
            dependency_graph=dag,
            min_capabilities={"skills": [ActionType.PICK, ActionType.DROP, ActionType.COOK, ActionType.BOIL, ActionType.CUT, ActionType.WASH, ActionType.ASSEMBLE], "min_robots": 3},
            parallel_opportunities=["Sink washing, cutting, stove boiling/cooking, and table assembly run in a synchronized 3-robot pipeline."]
        )

    @classmethod
    def get_all_tasks(cls) -> List[Task]:
        """Returns fresh instances of all 6 benchmark tasks in order (Task 1 through Task 6)."""
        return [
            cls.task_1_prepare_salad(),
            cls.task_2_prepare_sandwich_and_drink(),
            cls.task_3_prepare_hamburgers(),
            cls.task_4_prepare_soup_and_salad(),
            cls.task_5_prepare_two_hamburgers_and_salad(),
            cls.task_6_prepare_complete_meal(),
        ]

    @classmethod
    def get_task(cls, identifier: Union[str, int]) -> Optional[Task]:
        """
        Looks up and instantiates one of the 6 benchmark tasks by integer index (`1`–`6`),
        task ID (`"task_1"`–`"task_6"`), or keyword (`"salad"`, `"sandwich"`, `"hamburger"`, `"soup"`, `"meal"`).
        """
        tasks = {
            1: cls.task_1_prepare_salad,
            "1": cls.task_1_prepare_salad,
            "task_1": cls.task_1_prepare_salad,
            "salad": cls.task_1_prepare_salad,

            2: cls.task_2_prepare_sandwich_and_drink,
            "2": cls.task_2_prepare_sandwich_and_drink,
            "task_2": cls.task_2_prepare_sandwich_and_drink,
            "sandwich": cls.task_2_prepare_sandwich_and_drink,

            3: cls.task_3_prepare_hamburgers,
            "3": cls.task_3_prepare_hamburgers,
            "task_3": cls.task_3_prepare_hamburgers,
            "hamburger": cls.task_3_prepare_hamburgers,

            4: cls.task_4_prepare_soup_and_salad,
            "4": cls.task_4_prepare_soup_and_salad,
            "task_4": cls.task_4_prepare_soup_and_salad,
            "soup": cls.task_4_prepare_soup_and_salad,

            5: cls.task_5_prepare_two_hamburgers_and_salad,
            "5": cls.task_5_prepare_two_hamburgers_and_salad,
            "task_5": cls.task_5_prepare_two_hamburgers_and_salad,

            6: cls.task_6_prepare_complete_meal,
            "6": cls.task_6_prepare_complete_meal,
            "task_6": cls.task_6_prepare_complete_meal,
            "meal": cls.task_6_prepare_complete_meal,
        }
        factory = tasks.get(identifier if not isinstance(identifier, str) else identifier.lower())
        return factory() if factory else None
