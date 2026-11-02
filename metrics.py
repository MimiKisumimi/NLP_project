"""
Failure injection and objective metrics engine (`metrics.py`).

Implements seed-controlled failure and recovery scenarios (`FailureType`, `FailureTrigger`,
`FailureEvent`, `FailureScenarioSuite`, `FailureManager`) and simulator-grounded
task-performance and natural-language communication metrics (`TaskPerformanceMetrics`,
`CommunicationMetrics`, `MetricsEngine`).
"""

import re
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set, Union

from models import ActionType, CommunicationIntent, Item, Message


# ---------------------------------------------------------
# Seed-Controlled Failure Scenarios
# ---------------------------------------------------------
class FailureType(str, Enum):
    """The 5 standard controlled failure scenarios."""
    NONE = "none"
    TOOL_UNAVAILABLE = "tool_unavailable"             # A tool becomes temporarily unavailable (misplaced or unusable)
    WORKSTATION_BLOCKED = "workstation_blocked"       # A workstation is occupied / temporarily obstructed
    ACTION_EXECUTION_SLIP = "action_execution_slip"   # An attempted valid action fails with execution error
    MISPLACED_DELIVERY = "misplaced_delivery"         # An object is delivered / dropped to the wrong location
    MESSAGE_DELAY_OR_DROP = "message_delay_or_drop"   # A status message is delayed or omitted (packet loss/latency)

    @classmethod
    def from_string(cls, val: Any) -> Optional["FailureType"]:
        """Resolves an existing enum or matching string into a FailureType member."""
        if isinstance(val, cls):
            return val
        if isinstance(val, str):
            clean = val.strip().lower()
            try:
                return cls(clean)
            except ValueError:
                try:
                    return cls[val.strip().upper()]
                except KeyError:
                    pass
            for member in cls:
                if clean in member.value or clean in member.name.lower():
                    return member
        return None


@dataclass
class FailureTrigger:
    """Configuration for a deterministic or seed-controlled failure scenario."""
    trigger_id: str
    name: str
    failure_type: FailureType
    description: str
    trigger_timestep: int = 3                         # Timestep at which the failure activates or becomes armed
    duration: int = 4                                 # Duration in seconds (for tool unavailability, station block, or message delay)
    target_tool: Optional[str] = "knife"              # Tool name for TOOL_UNAVAILABLE
    target_station: Optional[str] = "Cutting table"   # Station name for WORKSTATION_BLOCKED
    wrong_station: Optional[str] = "Sink"             # Destination station for MISPLACED_DELIVERY
    drop_message: bool = False                        # True = omit message completely; False = delay message by `duration` seconds
    seed: int = 42
    triggered: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `FailureTrigger` configuration and status into a dictionary."""
        return {
            "trigger_id": self.trigger_id,
            "name": self.name,
            "failure_type": self.failure_type.value,
            "description": self.description,
            "trigger_timestep": self.trigger_timestep,
            "duration": self.duration,
            "target_tool": self.target_tool,
            "target_station": self.target_station,
            "wrong_station": self.wrong_station,
            "drop_message": self.drop_message,
            "seed": self.seed,
            "triggered": self.triggered,
            "resolved": self.resolved,
        }


@dataclass
class FailureEvent:
    """Records a triggered failure and tracks agent detection, communication, replanning, and recovery."""
    event_id: str
    failure_type: FailureType
    timestep_injected: int
    description: str
    affected_agent: Optional[str] = None
    affected_resource: Optional[str] = None
    detected: bool = False
    detected_at_timestep: Optional[int] = None
    communicated_cause: bool = False
    communicated_at_timestep: Optional[int] = None
    replanned: bool = False
    replanned_at_timestep: Optional[int] = None
    recovered: bool = False
    recovered_at_timestep: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `FailureEvent` lifecycle record into a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "failure_type": self.failure_type.value,
            "timestep_injected": self.timestep_injected,
            "description": self.description,
            "affected_agent": self.affected_agent,
            "affected_resource": self.affected_resource,
            "detected": self.detected,
            "detected_at_timestep": self.detected_at_timestep,
            "communicated_cause": self.communicated_cause,
            "communicated_at_timestep": self.communicated_at_timestep,
            "replanned": self.replanned,
            "replanned_at_timestep": self.replanned_at_timestep,
            "recovered": self.recovered,
            "recovered_at_timestep": self.recovered_at_timestep,
        }


class FailureScenarioSuite:
    """Provides the standard selectable failure scenarios."""

    @staticmethod
    def get_all_scenarios(seed: int = 42) -> List[FailureTrigger]:
        """Returns the 6 standard failure options (1 = No Failure, 2-6 = The 5 Failure Scenarios)."""
        return [
            FailureTrigger(
                trigger_id="failure_0",
                name="No Failure Injection (Clean Run)",
                failure_type=FailureType.NONE,
                description="Standard execution without any injected hardware, environment, or communication failures.",
                trigger_timestep=0,
                duration=0,
                seed=seed,
            ),
            FailureTrigger(
                trigger_id="failure_1",
                name="Tool Temporarily Unavailable",
                failure_type=FailureType.TOOL_UNAVAILABLE,
                description="A required tool (e.g. knife / pan) becomes temporarily unavailable/unusable at t=3s for 4s.",
                trigger_timestep=3,
                duration=4,
                target_tool="knife",
                seed=seed,
            ),
            FailureTrigger(
                trigger_id="failure_2",
                name="Workstation Occupied / Blocked",
                failure_type=FailureType.WORKSTATION_BLOCKED,
                description="A key workstation (e.g. Cutting table) is temporarily obstructed/occupied at t=3s for 4s.",
                trigger_timestep=3,
                duration=4,
                target_station="Cutting table",
                seed=seed,
            ),
            FailureTrigger(
                trigger_id="failure_3",
                name="Attempted Action Fails (Execution Slip)",
                failure_type=FailureType.ACTION_EXECUTION_SLIP,
                description="An attempted valid physical action (CUT, COOK, BOIL, WASH, PICK, or ASSEMBLE) fails with a gripper/execution error at t>=3s.",
                trigger_timestep=3,
                duration=1,
                seed=seed,
            ),
            FailureTrigger(
                trigger_id="failure_4",
                name="Object Delivered to Wrong Location",
                failure_type=FailureType.MISPLACED_DELIVERY,
                description="When an agent attempts to DROP/deliver an item at t>=3s, the item slips and is placed at the wrong workstation (Sink).",
                trigger_timestep=3,
                duration=1,
                wrong_station="Sink",
                seed=seed,
            ),
            FailureTrigger(
                trigger_id="failure_5",
                name="Status Message Delayed or Omitted",
                failure_type=FailureType.MESSAGE_DELAY_OR_DROP,
                description="An inter-agent status/coordination message sent at t>=2s is intercepted and delayed (or omitted), testing communication recovery.",
                trigger_timestep=2,
                duration=3,
                drop_message=False,
                seed=seed,
            ),
        ]

    @classmethod
    def get_scenario_by_option(cls, option: Union[int, str], seed: int = 42) -> List[FailureTrigger]:
        """
        Resolves user menu option (1-7 or name) into a list of FailureTrigger objects:
        1: None
        2: Tool Temporarily Unavailable
        3: Workstation Occupied / Blocked
        4: Attempted Action Fails (Execution Slip)
        5: Object Delivered to Wrong Location
        6: Status Message Delayed or Omitted
        7: Combined Medium/Hard Stress Suite (Multiple seed-controlled failures)
        """
        scenarios = cls.get_all_scenarios(seed=seed)
        if isinstance(option, int):
            if option == 1:
                return []
            if 2 <= option <= 6:
                return [scenarios[option - 1]]
            if option == 7:
                # Combined multi-failure suite keyed by seed
                return [
                    FailureTrigger(
                        trigger_id="combo_tool",
                        name="Tool Temporarily Unavailable",
                        failure_type=FailureType.TOOL_UNAVAILABLE,
                        description="Knife becomes temporarily unavailable at t=2s for 3s.",
                        trigger_timestep=2,
                        duration=3,
                        target_tool="knife",
                        seed=seed,
                    ),
                    FailureTrigger(
                        trigger_id="combo_slip",
                        name="Attempted Action Execution Slip",
                        failure_type=FailureType.ACTION_EXECUTION_SLIP,
                        description="First manipulation action at t>=5s suffers an execution slip.",
                        trigger_timestep=5,
                        duration=1,
                        seed=seed,
                    ),
                    FailureTrigger(
                        trigger_id="combo_msg",
                        name="Status Message Delayed",
                        failure_type=FailureType.MESSAGE_DELAY_OR_DROP,
                        description="Peer message at t>=3s is delayed by 2s.",
                        trigger_timestep=3,
                        duration=2,
                        drop_message=False,
                        seed=seed,
                    ),
                ]
        if isinstance(option, str):
            clean = option.strip().lower()
            if clean in ("none", "0", "1", "no"):
                return []
            if clean in ("all", "combined", "7", "multi"):
                return cls.get_scenario_by_option(7, seed=seed)
            ft = FailureType.from_string(clean)
            for sc in scenarios[1:]:
                if sc.failure_type == ft:
                    return [sc]
        return []


class FailureManager:
    """
    Manages deterministic/seed-controlled failure injection and tracks agent detection,
    failure communication, plan revision, and recovery.
    """

    def __init__(self, triggers: Optional[List[FailureTrigger]] = None, seed: int = 42):
        """Initializes the `FailureManager` with a deterministic random seed and list of `FailureTrigger` items."""
        self.seed: int = seed
        self.rng = random.Random(seed)
        self.triggers: List[FailureTrigger] = [
            t for t in (triggers or []) if t.failure_type != FailureType.NONE
        ]
        self.events: List[FailureEvent] = []
        self.delayed_messages: List[Tuple[int, Message]] = []  # List of (release_timestep, Message)

    def step_pre_actions(self, env: Any, timestep: int):
        """
        Called at the start of `env.timestep(timestep)` to:
        1. Release any delayed messages whose release_timestep <= timestep.
        2. Inject time-based environmental failures (TOOL_UNAVAILABLE, WORKSTATION_BLOCKED).
        """
        # 1. Deliver delayed messages that have matured
        remaining_delayed: List[Tuple[int, Message]] = []
        for release_t, msg in self.delayed_messages:
            if timestep >= release_t:
                print(f"[FailureManager] 📨 Delivering delayed message from {msg.sender} (originally t={msg.timestep}s) at t={timestep}s.")
                env._deliver_routed_message(msg)
            else:
                remaining_delayed.append((release_t, msg))
        self.delayed_messages = remaining_delayed

        # Check time-triggered environment locks (Tool Unavailable / Workstation Blocked)
        for trig in self.triggers:
            if trig.triggered:
                continue

            if trig.failure_type == FailureType.TOOL_UNAVAILABLE and timestep >= trig.trigger_timestep:
                # Find target tool or fallback to first tool in environment
                tool_item = None
                for it in env.items.values():
                    if it.is_tool and (not trig.target_tool or trig.target_tool.lower() in it.name.lower()):
                        tool_item = it
                        break
                if not tool_item:
                    tool_item = next((it for it in env.items.values() if it.is_tool), None)

                if tool_item:
                    trig.triggered = True
                    until_t = timestep + trig.duration
                    env.tool_locks[tool_item.id] = {
                        "locked_by": "FAILURE_INJECTION",
                        "tool_name": tool_item.name,
                        "until_timestep": until_t,
                        "action": "TOOL_UNAVAILABLE",
                    }
                    desc = (
                        f"FAILURE INJECTED [Tool Unavailable]: Tool '{tool_item.name}' ({tool_item.id}) "
                        f"at {tool_item.location} is misplaced/unavailable until t={until_t}s."
                    )
                    print(f"[FailureManager] 💥 {desc} (Silent perturbation - awaiting agent detection)")
                    event = FailureEvent(
                        event_id=f"ev_{trig.trigger_id}_t{timestep}",
                        failure_type=FailureType.TOOL_UNAVAILABLE,
                        timestep_injected=timestep,
                        description=desc,
                        affected_resource=tool_item.name,
                        detected=False,
                        detected_at_timestep=None,
                    )
                    self.events.append(event)

            elif trig.failure_type == FailureType.WORKSTATION_BLOCKED and timestep >= trig.trigger_timestep:
                target_ws = trig.target_station or "Cutting table"
                ws_obj = env.find_workstation(target_ws)
                ws_name = ws_obj.name if ws_obj else "Cutting table"

                trig.triggered = True
                until_t = timestep + trig.duration
                env.workstation_locks[ws_name] = {
                    "locked_by": "FAILURE_INJECTION",
                    "until_timestep": until_t,
                    "action": "WORKSTATION_BLOCKED",
                }
                desc = (
                    f"FAILURE INJECTED [Workstation Occupied/Blocked]: Workstation '{ws_name}' "
                    f"is obstructed/occupied until t={until_t}s."
                )
                print(f"[FailureManager] 💥 {desc} (Silent perturbation - awaiting agent detection)")
                event = FailureEvent(
                    event_id=f"ev_{trig.trigger_id}_t{timestep}",
                    failure_type=FailureType.WORKSTATION_BLOCKED,
                    timestep_injected=timestep,
                    description=desc,
                    affected_resource=ws_name,
                    detected=False,
                    detected_at_timestep=None,
                )
                self.events.append(event)

    def intercept_message(self, env: Any, message: Message, timestep: int) -> bool:
        """
        Checks if an outgoing peer message should be delayed or dropped (FailureType.MESSAGE_DELAY_OR_DROP).
        Returns True if intercepted (caller should NOT deliver immediately), False otherwise.
        """
        if message.sender == "System":
            return False

        # First inspect every outgoing peer message for failure detection / communication / replanning tracking
        self.observe_peer_message(message, timestep)

        for trig in self.triggers:
            if not trig.triggered and trig.failure_type == FailureType.MESSAGE_DELAY_OR_DROP and timestep >= trig.trigger_timestep:
                trig.triggered = True
                mode_str = "OMITTED/DROPPED" if trig.drop_message else f"DELAYED by {trig.duration}s (until t={timestep + trig.duration}s)"
                desc = (
                    f"FAILURE INJECTED [Message {mode_str}]: Status message from {message.sender} "
                    f"to {message.recipient} at t={timestep}s was {mode_str}."
                )
                print(f"[FailureManager] 💥 {desc}")
                event = FailureEvent(
                    event_id=f"ev_{trig.trigger_id}_t{timestep}",
                    failure_type=FailureType.MESSAGE_DELAY_OR_DROP,
                    timestep_injected=timestep,
                    description=desc,
                    affected_agent=message.sender,
                    affected_resource="communication_channel",
                    detected=True,
                    detected_at_timestep=timestep,
                )
                self.events.append(event)

                if not trig.drop_message:
                    self.delayed_messages.append((timestep + trig.duration, message))

                # Notify sender that a communication glitch occurred so detection/recovery can be measured
                if message.sender in env.agents:
                    env.agents[message.sender].receive_message(Message(
                        sender="System",
                        recipient=message.sender,
                        timestep=timestep,
                        function=ActionType.ERROR,
                        intent=CommunicationIntent.WARNING,
                        content=f"Communication Notice: Your message at t={timestep}s experienced network interference ({mode_str}).",
                    ))
                return True
        return False

    def intercept_action_execution(
        self,
        env: Any,
        agent: Any,
        action: Message,
        action_type: Optional[ActionType],
        timestep: int
    ) -> Tuple[bool, str]:
        """
        Checks if a valid physical action should fail due to ACTION_EXECUTION_SLIP.
        Returns (should_fail, failure_error_message).
        """
        if action_type in (None, ActionType.WAIT, ActionType.MOVE, ActionType.MESSAGE):
            return False, ""

        for trig in self.triggers:
            if not trig.triggered and trig.failure_type == FailureType.ACTION_EXECUTION_SLIP and timestep >= trig.trigger_timestep:
                trig.triggered = True
                func_str = action_type.value if action_type else str(action.function)
                desc = (
                    f"FAILURE INJECTED [Attempted Action Fails]: {agent.robot_id}'s valid action "
                    f"'{func_str}' failed at t={timestep}s due to an unexpected mechanical slip."
                )
                print(f"[FailureManager] 💥 {desc}")
                event = FailureEvent(
                    event_id=f"ev_{trig.trigger_id}_t{timestep}",
                    failure_type=FailureType.ACTION_EXECUTION_SLIP,
                    timestep_injected=timestep,
                    description=desc,
                    affected_agent=agent.robot_id,
                    affected_resource=func_str,
                    detected=True,
                    detected_at_timestep=timestep,
                )
                self.events.append(event)
                return True, f"Execution Failure (Injected Slip): Attempted action '{func_str}' failed unexpectedly! Communicate with peers and retry or revise plan."
        return False, ""

    def intercept_drop_completion(self, env: Any, agent: Any, act: Dict[str, Any], timestep: int) -> Optional[str]:
        """
        Checks if a completing DROP action should misplace the item at the wrong station (MISPLACED_DELIVERY).
        Returns the overridden wrong station name if triggered, else None.
        """
        for trig in self.triggers:
            if not trig.triggered and trig.failure_type == FailureType.MISPLACED_DELIVERY and timestep >= trig.trigger_timestep:
                intended_ws = act.get("station") or "Food assembly table"
                wrong_ws = trig.wrong_station or "Sink"
                if wrong_ws == intended_ws:
                    wrong_ws = "Refrigerator"

                trig.triggered = True
                target_item = act.get("target_item") or "item"
                desc = (
                    f"FAILURE INJECTED [Misplaced Delivery]: {agent.robot_id} attempted to deliver "
                    f"'{target_item}' to '{intended_ws}', but it was misplaced at '{wrong_ws}' at t={timestep}s!"
                )
                print(f"[FailureManager] 💥 {desc} (Silent perturbation - awaiting agent detection)")
                event = FailureEvent(
                    event_id=f"ev_{trig.trigger_id}_t{timestep}",
                    failure_type=FailureType.MISPLACED_DELIVERY,
                    timestep_injected=timestep,
                    description=desc,
                    affected_agent=agent.robot_id,
                    affected_resource=target_item,
                    detected=False,
                    detected_at_timestep=None,
                )
                self.events.append(event)
                return wrong_ws
        return None

    def observe_environment_state(self, env: Any, timestep: int):
        """Checks if co-located agents detect active environmental failure locks through local observation."""
        for ev in self.events:
            if not ev.detected:
                if ev.failure_type == FailureType.WORKSTATION_BLOCKED:
                    for r_id, ag in getattr(env, "agents", {}).items():
                        if getattr(ag, "location", "") == ev.affected_resource:
                            ev.detected = True
                            ev.detected_at_timestep = timestep
                            ev.affected_agent = r_id
                            break
                elif ev.failure_type == FailureType.TOOL_UNAVAILABLE:
                    for r_id, ag in getattr(env, "agents", {}).items():
                        loc = getattr(ag, "location", "")
                        for it in getattr(env, "items", {}).values():
                            if (
                                it.is_tool
                                and (ev.affected_resource or "").lower() in it.name.lower()
                                and it.location == loc
                            ):
                                ev.detected = True
                                ev.detected_at_timestep = timestep
                                ev.affected_agent = r_id
                                break

    def observe_action_rejection(self, robot_id: str, error_msg: str, timestep: int):
        """Marks an active failure as detected if an agent hits a failure lock or rejection."""
        err_lower = error_msg.lower()
        for ev in self.events:
            if not ev.detected:
                is_match = False
                res_name = (ev.affected_resource or "").lower()
                if ev.failure_type == FailureType.TOOL_UNAVAILABLE:
                    if (
                        "failure_injection" in err_lower
                        or "unavailable" in err_lower
                        or "locked" in err_lower
                        or (res_name and res_name in err_lower)
                    ):
                        is_match = True
                elif ev.failure_type == FailureType.WORKSTATION_BLOCKED:
                    if (
                        "failure_injection" in err_lower
                        or "blocked" in err_lower
                        or "occupied" in err_lower
                        or (res_name and res_name in err_lower)
                    ):
                        is_match = True
                elif ev.failure_type == FailureType.MISPLACED_DELIVERY:
                    if (
                        "no matching item" in err_lower
                        or "misplaced" in err_lower
                        or (res_name and res_name in err_lower)
                    ):
                        is_match = True
                elif ev.failure_type == FailureType.ACTION_EXECUTION_SLIP:
                    if "slip" in err_lower or "execution failure" in err_lower or (res_name and res_name in err_lower):
                        is_match = True

                if is_match:
                    ev.detected = True
                    ev.detected_at_timestep = timestep
                    if not ev.affected_agent:
                        ev.affected_agent = robot_id

    def observe_peer_message(self, message: Message, timestep: int):
        """
        Evaluates whether a peer message communicates a failure cause or proposes an alternative plan
        following an injected failure.
        """
        if not self.events or message.sender == "System":
            return

        text = message.message_content.lower()
        intent = message.intent

        failure_keywords = (
            "fail", "error", "unavailable", "blocked", "occupied", "locked",
            "wrong", "misplaced", "slip", "delay", "wait", "cannot", "can't", "obstruction"
        )
        replan_keywords = (
            "instead", "alternative", "replan", "revise", "switch", "meanwhile",
            "other", "take over", "help", "fetch", "retry", "will do", "propose"
        )

        is_cause_comm = (
            intent in (CommunicationIntent.FAILURE_EXPLANATION, CommunicationIntent.WARNING, CommunicationIntent.CONFLICT_RESOLUTION)
            or any(kw in text for kw in failure_keywords)
        )
        is_replan_comm = (
            intent in (CommunicationIntent.PLAN_REVISION, CommunicationIntent.PROPOSAL, CommunicationIntent.REQUEST, CommunicationIntent.COMMITMENT)
            or any(kw in text for kw in replan_keywords)
        )

        for ev in self.events:
            if timestep >= ev.timestep_injected:
                if is_cause_comm:
                    if not ev.detected:
                        ev.detected = True
                        ev.detected_at_timestep = timestep
                        if not ev.affected_agent:
                            ev.affected_agent = message.sender
                    if not ev.communicated_cause:
                        ev.communicated_cause = True
                        ev.communicated_at_timestep = timestep
                if is_replan_comm and not ev.replanned:
                    ev.replanned = True
                    ev.replanned_at_timestep = timestep

    def evaluate_recovery(self, env: Any, timestep: int):
        """
        Updates recovery status on all triggered failure events.
        A failure event is marked recovered if:
        - Subsequent actions succeed after the failure timestamp, or
        - The overall task is completed afterward.
        """
        self.observe_environment_state(env, timestep)
        task_done = env.is_task_complete()
        for ev in self.events:
            if ev.recovered:
                continue
            # Check if any physical action completed after the failure was injected
            post_actions = [
                ca for ca in env.completed_actions
                if ca.get("timestep", 0) > ev.timestep_injected
            ]
            if task_done or len(post_actions) > 0:
                ev.recovered = True
                ev.recovered_at_timestep = timestep

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregates all injected `FailureEvent` instances into summary counts and rates
        for detection, cause communication, plan revision, and post-failure recovery.
        """
        total = len(self.events)
        detected = sum(1 for e in self.events if e.detected)
        communicated = sum(1 for e in self.events if e.communicated_cause)
        replanned = sum(1 for e in self.events if e.replanned)
        recovered = sum(1 for e in self.events if e.recovered)
        return {
            "configured_triggers": [t.to_dict() for t in self.triggers],
            "total_failures_injected": total,
            "failures_detected": detected,
            "failures_communicated": communicated,
            "plans_revised": replanned,
            "failures_recovered": recovered,
            "detection_rate": round(detected / total, 3) if total > 0 else 1.0,
            "communication_rate": round(communicated / total, 3) if total > 0 else 1.0,
            "replanning_rate": round(replanned / total, 3) if total > 0 else 1.0,
            "recovery_rate": round(recovered / total, 3) if total > 0 else 1.0,
            "events": [e.to_dict() for e in self.events],
        }


# ---------------------------------------------------------
# Task-Performance and Coordination Metrics
# ---------------------------------------------------------
@dataclass
class TaskPerformanceMetrics:
    """Objective task-performance and coordination metrics."""
    task_name: str
    condition_name: str
    task_success: bool                                # Ground-truth simulator verification
    task_success_rate: float                          # 1.0 if complete else 0.0 (for single run; aggregated across runs)
    valid_action_rate: float                          # Valid physical actions / total attempted actions
    constraint_violation_rate: float                  # Rejected/invalid actions / total attempted actions
    makespan: int                                     # Total completion time (or max timesteps elapsed)
    total_actions: int                                # Total attempted actions (completed + failed)
    completed_actions_count: int                      # Total valid completed physical actions
    failed_actions_count: int                         # Total rejected/invalid actions
    total_movements: int                              # Total MOVE actions executed
    idle_timesteps: int                               # Total agent-seconds spent waiting/idle
    resource_conflicts: int                           # Mutual exclusion contention count
    dependency_errors: int                            # Out-of-order / unmet precondition attempts
    replanning_frequency: int                         # Count of PLAN_REVISION / FAILURE_EXPLANATION / adaptive replans
    recovery_rate: float                              # Post-failure/rejection recovery rate
    workload_distribution: Dict[str, int]             # Completed actions per robot_id
    workload_shares: Dict[str, float]                 # Proportion of work per robot_id
    # ###Legacy metrics that were not used in the final analysis###
    workload_gini: float                              # Gini coefficient of workload balance (0.0 = perfect balance)
    failure_recovery_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes `TaskPerformanceMetrics` into a JSON-compatible dictionary."""
        return {
            "task_name": self.task_name,
            "condition_name": self.condition_name,
            "task_success": self.task_success,
            "task_success_rate": self.task_success_rate,
            "valid_action_rate": self.valid_action_rate,
            "constraint_violation_rate": self.constraint_violation_rate,
            "makespan": self.makespan,
            "total_actions": self.total_actions,
            "completed_actions_count": self.completed_actions_count,
            "failed_actions_count": self.failed_actions_count,
            "total_movements": self.total_movements,
            "idle_timesteps": self.idle_timesteps,
            "resource_conflicts": self.resource_conflicts,
            "dependency_errors": self.dependency_errors,
            "replanning_frequency": self.replanning_frequency,
            "recovery_rate": self.recovery_rate,
            "workload_distribution": self.workload_distribution,
            "workload_shares": self.workload_shares,
            "workload_gini": self.workload_gini,
            "failure_recovery_summary": self.failure_recovery_summary,
        }


# ---------------------------------------------------------
# Natural-Language Communication Metrics
# ---------------------------------------------------------
@dataclass
class CommunicationMetrics:
    """Objective natural-language communication metrics."""
    number_of_messages: int                           # Total peer-to-peer messages sent
    total_communication_tokens: int                   # Total word tokens across all peer messages
    messages_per_successful_task: Optional[float]     # Messages if task succeeded (else None)
    tokens_per_successful_task: Optional[float]       # Tokens if task succeeded (else None)
    average_message_length_tokens: float              # Mean words/tokens per peer message
    # ###Legacy metrics that were not used in the final analysis###
    average_message_length_chars: float               # Mean characters per peer message
    grounded_message_rate: float                      # Proportion of messages consistent with simulator ground truth
    ungrounded_messages_count: int                    # Count of messages contradicting simulator state
    ungrounded_examples: List[str]                    # Diagnostic descriptions of ungrounded claims
    redundant_message_rate: float                     # Proportion of duplicate/repetitive messages
    redundant_messages_count: int                     # Count of redundant messages
    communication_function_distribution: Dict[str, float]  # Percentage breakdown across the 8 speech acts
    communication_function_counts: Dict[str, int]     # Raw counts across the 8 speech acts
    # ###Legacy metrics that were not used in the final analysis###
    comm_volume_to_progress_ratio: float              # Messages per completed DAG step (relationship between comm volume & success)
    llm_judge_scores: Optional[Dict[str, int]] = None # 1-5 scores from Independent LLM Judge across the 7 dimensions
    llm_judge_overall_score: Optional[float] = None   # Mean 1-5 score across the 7 LLM Judge dimensions

    def to_dict(self) -> Dict[str, Any]:
        """Serializes `CommunicationMetrics` into a JSON-compatible dictionary."""
        return {
            "number_of_messages": self.number_of_messages,
            "total_communication_tokens": self.total_communication_tokens,
            "messages_per_successful_task": self.messages_per_successful_task,
            "tokens_per_successful_task": self.tokens_per_successful_task,
            "average_message_length_tokens": self.average_message_length_tokens,
            "average_message_length_chars": self.average_message_length_chars,
            "grounded_message_rate": self.grounded_message_rate,
            "ungrounded_messages_count": self.ungrounded_messages_count,
            "ungrounded_examples": self.ungrounded_examples,
            "redundant_message_rate": self.redundant_message_rate,
            "redundant_messages_count": self.redundant_messages_count,
            "communication_function_distribution": self.communication_function_distribution,
            "communication_function_counts": self.communication_function_counts,
            "comm_volume_to_progress_ratio": self.comm_volume_to_progress_ratio,
            "llm_judge_scores": self.llm_judge_scores,
            "llm_judge_overall_score": self.llm_judge_overall_score,
        }


# ---------------------------------------------------------
# Unified Metrics Calculation Engine
# ---------------------------------------------------------
class MetricsEngine:
    """Computes objective Task Performance and Communication metrics from Environment state."""

    # ###Legacy metrics that were not used in the final analysis###
    @staticmethod
    def _compute_gini(counts: List[int]) -> float:
        """Computes the Gini coefficient (0.0 = perfectly equal workload, 1.0 = maximally unequal)."""
        if not counts or sum(counts) == 0 or len(counts) <= 1:
            return 0.0
        sorted_c = sorted(counts)
        n = len(sorted_c)
        total = sum(sorted_c)
        cum_sum = sum((i + 1) * val for i, val in enumerate(sorted_c))
        gini = (2.0 * cum_sum) / (n * total) - (n + 1.0) / n
        return round(max(0.0, min(1.0, gini)), 3)

    @classmethod
    def compute_task_metrics(cls, env: Any) -> TaskPerformanceMetrics:
        """Calculates all Task-Performance and Coordination Metrics."""
        verif = env.verify_current_task()
        task_success = bool(verif.is_complete)
        task_success_rate = 1.0 if task_success else 0.0

        completed_list = list(getattr(env, "completed_actions", []))
        failed_list = list(getattr(env, "failed_actions", []))
        conflicts_list = list(getattr(env, "resource_conflicts", []))

        completed_count = len(completed_list)
        failed_count = len(failed_list)
        total_actions = completed_count + failed_count

        valid_action_rate = round(completed_count / total_actions, 3) if total_actions > 0 else 1.0
        constraint_violation_rate = round(failed_count / total_actions, 3) if total_actions > 0 else 0.0

        makespan = int(getattr(env, "current_time", len(getattr(env, "state_history", []))))

        # Count total movements (MOVE actions completed)
        total_movements = sum(
            1 for ca in completed_list
            if ca.get("action", {}).get("function") == ActionType.MOVE.value
        )

        # Count idle timesteps across all agents
        # An agent is idle in a timestep if it executed WAIT or took no physical action while not busy
        num_agents = max(1, len(getattr(env, "agents", {})))
        total_agent_seconds = makespan * num_agents
        busy_agent_seconds = sum(
            int(ca.get("action", {}).get("duration", 1))
            for ca in completed_list
            if ca.get("action", {}).get("function") != ActionType.WAIT.value
        )
        idle_timesteps = max(0, total_agent_seconds - busy_agent_seconds)

        # Count dependency errors (attempting ASSEMBLE, CUT, COOK, BOIL before required ingredients/tools/states are ready)
        dep_keywords = (
            "requires", "already cut", "no cuttable", "no matching item",
            "missing", "not carrying", "no ingredients", "no item found"
        )
        dependency_errors = sum(
            1 for fa in failed_list
            if any(kw in str(fa.get("error", "")).lower() for kw in dep_keywords)
        )

        # Replanning frequency: count of PLAN_REVISION or FAILURE_EXPLANATION messages + adaptive action switches after failure
        peer_msgs: List[Message] = list(getattr(env, "peer_message_log", []))
        replan_msgs = sum(
            1 for m in peer_msgs
            if m.intent in (CommunicationIntent.PLAN_REVISION, CommunicationIntent.FAILURE_EXPLANATION, CommunicationIntent.CONFLICT_RESOLUTION)
        )

        # Check failure manager summary if present
        failure_mgr: Optional[FailureManager] = getattr(env, "failure_manager", None)
        if failure_mgr and failure_mgr.events:
            failure_mgr.evaluate_recovery(env, makespan)
            fm_summary = failure_mgr.get_summary()
            recovery_rate = fm_summary["recovery_rate"]
            replanning_frequency = max(replan_msgs, fm_summary["plans_revised"])
        else:
            fm_summary = {}
            replanning_frequency = replan_msgs
            # If there were action rejections, measure whether agents recovered to complete subsequent valid actions
            if failed_count > 0:
                recovered_agents = 0
                agents_with_failures = {fa.get("robot_id") for fa in failed_list if fa.get("robot_id")}
                for r_id in agents_with_failures:
                    first_fail_t = min(fa.get("timestep", 0) for fa in failed_list if fa.get("robot_id") == r_id)
                    if any(ca.get("robot_id") == r_id and ca.get("timestep", 0) > first_fail_t for ca in completed_list):
                        recovered_agents += 1
                recovery_rate = round(recovered_agents / len(agents_with_failures), 3) if agents_with_failures else 1.0
            else:
                recovery_rate = 1.0

        # Workload distribution across agents
        workload_dist: Dict[str, int] = {r_id: 0 for r_id in getattr(env, "agents", {}).keys()}
        for ca in completed_list:
            r_id = ca.get("robot_id")
            if r_id:
                workload_dist[r_id] = workload_dist.get(r_id, 0) + 1

        total_completed = sum(workload_dist.values())
        workload_shares = {
            r_id: round(cnt / total_completed, 3) if total_completed > 0 else 0.0
            for r_id, cnt in workload_dist.items()
        }
        # ###Legacy metrics that were not used in the final analysis###
        workload_gini = cls._compute_gini(list(workload_dist.values()))

        task_name = env.current_task.name if getattr(env, "current_task", None) else "Unknown"
        cond_name = env.condition_config.name if getattr(env, "condition_config", None) else "Unknown"

        return TaskPerformanceMetrics(
            task_name=task_name,
            condition_name=cond_name,
            task_success=task_success,
            task_success_rate=task_success_rate,
            valid_action_rate=valid_action_rate,
            constraint_violation_rate=constraint_violation_rate,
            makespan=makespan,
            total_actions=total_actions,
            completed_actions_count=completed_count,
            failed_actions_count=failed_count,
            total_movements=total_movements,
            idle_timesteps=idle_timesteps,
            resource_conflicts=len(conflicts_list),
            dependency_errors=dependency_errors,
            replanning_frequency=replanning_frequency,
            recovery_rate=recovery_rate,
            workload_distribution=workload_dist,
            workload_shares=workload_shares,
            workload_gini=workload_gini,
            failure_recovery_summary=fm_summary,
        )

    @classmethod
    def _check_message_grounding(cls, msg: Message, env: Any) -> Tuple[bool, str]:
        """
        Deterministically checks whether a peer message is factually grounded in the simulator state.
        A message is ungrounded when it contradicts the environment state — e.g., claiming an ingredient
        has been cut, cooked, boiled, washed, or served when the simulator records it otherwise.
        """
        text = msg.message_content.strip().lower()
        if not text:
            return True, ""

        # Retrieve simulator state at message timestep (or fallback to current items)
        state_at_t: Optional[Dict[str, Any]] = None
        for snap in getattr(env, "state_history", []):
            if snap.get("timestep") == msg.timestep:
                state_at_t = snap
                break

        items_map: Dict[str, Dict[str, Any]] = {}
        if state_at_t and "items" in state_at_t:
            items_map = state_at_t["items"]
        else:
            items_map = {k: it.to_dict() for k, it in getattr(env, "items", {}).items()}

        # Check factual past-tense completion claims ("I have cut tomato", "tomato is cut", "patty is cooked", etc.)
        # Avoid flagging future commitments ("I will cut tomato", "going to cut tomato")
        future_markers = ("will ", "going to ", "plan to ", "can ", "should ", "need to ", "let's ", "please ", "Heading ")
        if any(fm.lower() in text for fm in future_markers):
            return True, ""

        known_ingredients = {
            it_data.get("name", "").lower()
            for it_data in items_map.values()
            if not it_data.get("is_tool", False)
        }

        state_claim_patterns = [
            ("cut", r"\b(?:is|has been|finished|done)\s+cut(?:ting)?\b|\bi\s+(?:cut|have cut|sliced)\b"),
            ("cooked", r"\b(?:is|has been|finished)\s+cook(?:ed|ing)\b|\bi\s+(?:cooked|have cooked)\b"),
            ("boiled", r"\b(?:is|has been|finished)\s+boil(?:ed|ing)\b|\bi\s+(?:boiled|have boiled)\b"),
            ("washed", r"\b(?:is|has been|finished)\s+wash(?:ed|ing)\b|\bi\s+(?:washed|have washed)\b"),
            ("served", r"\b(?:is|has been)\s+served\b|\bi\s+(?:served|have served|delivered)\b"),
        ]

        for claimed_state, pattern in state_claim_patterns:
            if re.search(pattern, text):
                for ing_name in known_ingredients:
                    if ing_name and ing_name in text:
                        matching_objs = [
                            it_d for it_d in items_map.values()
                            if it_d.get("name", "").lower() == ing_name
                        ]
                        if matching_objs:
                            valid_states = {claimed_state, "assembled", "served"}
                            if not any(it_d.get("state", "").lower() in valid_states for it_d in matching_objs):
                                actual_states = [it_d.get("state", "raw") for it_d in matching_objs]
                                return (
                                    False,
                                    f"T{msg.timestep}s {msg.sender} claimed '{ing_name}' is '{claimed_state}', "
                                    f"but simulator records state={actual_states}."
                                )

        return True, ""

    @classmethod
    def compute_communication_metrics(cls, env: Any) -> CommunicationMetrics:
        """Calculates all Natural-Language Communication Metrics."""
        peer_msgs: List[Message] = list(getattr(env, "peer_message_log", []))
        num_msgs = len(peer_msgs)

        verif = env.verify_current_task()
        task_success = bool(verif.is_complete)

        # Token & character counts
        token_counts = [len(m.message_content.split()) for m in peer_msgs]
        # ###Legacy metrics that were not used in the final analysis###
        char_counts = [len(m.message_content) for m in peer_msgs]
        total_tokens = sum(token_counts)

        avg_tokens = round(total_tokens / num_msgs, 2) if num_msgs > 0 else 0.0
        # ###Legacy metrics that were not used in the final analysis###
        avg_chars = round(sum(char_counts) / num_msgs, 2) if num_msgs > 0 else 0.0

        msgs_per_success = float(num_msgs) if task_success else None
        tokens_per_success = float(total_tokens) if task_success else None

        # Grounded message rate & ungrounded detection
        ungrounded_count = 0
        ungrounded_examples: List[str] = []
        for m in peer_msgs:
            is_grounded, reason = cls._check_message_grounding(m, env)
            if not is_grounded:
                ungrounded_count += 1
                ungrounded_examples.append(reason)

        grounded_rate = round((num_msgs - ungrounded_count) / num_msgs, 3) if num_msgs > 0 else 1.0

        # Redundant message rate: exact or near-identical consecutive/repeated content from same sender
        seen_sender_texts: Set[Tuple[str, str]] = set()
        redundant_count = 0
        for m in peer_msgs:
            norm_text = re.sub(r"\s+", " ", m.message_content.strip().lower())
            key = (m.sender, norm_text)
            if norm_text and key in seen_sender_texts:
                redundant_count += 1
            elif norm_text:
                seen_sender_texts.add(key)

        redundant_rate = round(redundant_count / num_msgs, 3) if num_msgs > 0 else 0.0

        # Communication-function distribution across the 8 required intents
        intent_counts: Dict[str, int] = {intent.value: 0 for intent in CommunicationIntent}
        for m in peer_msgs:
            intent_key = (
                m.intent.value
                if isinstance(m.intent, CommunicationIntent)
                else (str(m.intent) if m.intent in intent_counts else CommunicationIntent.STATUS_UPDATE.value)
            )
            intent_counts[intent_key] = intent_counts.get(intent_key, 0) + 1

        intent_dist: Dict[str, float] = {
            k: round((v / num_msgs) * 100.0, 1) if num_msgs > 0 else 0.0
            for k, v in intent_counts.items()
        }

        # ###Legacy metrics that were not used in the final analysis###
        completed_nodes = (
            env.current_task.get_progress().get("completed_nodes", 0)
            if getattr(env, "current_task", None) else 0
        )
        comm_to_progress = round(num_msgs / max(1, completed_nodes), 2)

        return CommunicationMetrics(
            number_of_messages=num_msgs,
            total_communication_tokens=total_tokens,
            messages_per_successful_task=msgs_per_success,
            tokens_per_successful_task=tokens_per_success,
            average_message_length_tokens=avg_tokens,
            average_message_length_chars=avg_chars,
            grounded_message_rate=grounded_rate,
            ungrounded_messages_count=ungrounded_count,
            ungrounded_examples=ungrounded_examples,
            redundant_message_rate=redundant_rate,
            redundant_messages_count=redundant_count,
            communication_function_distribution=intent_dist,
            communication_function_counts=intent_counts,
            comm_volume_to_progress_ratio=comm_to_progress,
        )

    @classmethod
    def format_metrics_report(
        cls,
        task_metrics: TaskPerformanceMetrics,
        comm_metrics: CommunicationMetrics
    ) -> str:
        """Formats a clean console report of task, failure, and communication metrics."""
        lines = [
            "=" * 70,
            "OBJECTIVE PERFORMANCE & COMMUNICATION METRICS REPORT",
            "=" * 70,
            f"1. TASK-PERFORMANCE & COORDINATION METRICS:",
            f"   - Task Success (Ground Truth):  {task_metrics.task_success} (Rate: {task_metrics.task_success_rate:.2f})",
            f"   - Makespan (Completion Time):   {task_metrics.makespan}s",
            f"   - Valid Action Rate:            {task_metrics.valid_action_rate * 100:.1f}% ({task_metrics.completed_actions_count}/{task_metrics.total_actions})",
            f"   - Constraint Violation Rate:    {task_metrics.constraint_violation_rate * 100:.1f}% ({task_metrics.failed_actions_count}/{task_metrics.total_actions})",
            f"   - Total Movements (MOVE):       {task_metrics.total_movements}",
            f"   - Idle Timesteps (Agent-s):     {task_metrics.idle_timesteps}s",
            f"   - Resource Conflicts:           {task_metrics.resource_conflicts}",
            f"   - Dependency Errors:            {task_metrics.dependency_errors}",
            f"   - Replanning Frequency:         {task_metrics.replanning_frequency}",
            f"   - Recovery Rate:                {task_metrics.recovery_rate * 100:.1f}%",
            f"   - Workload Distribution:        {task_metrics.workload_distribution} (Gini: {task_metrics.workload_gini:.3f})",
        ]

        if task_metrics.failure_recovery_summary and task_metrics.failure_recovery_summary.get("total_failures_injected", 0) > 0:
            fm = task_metrics.failure_recovery_summary
            lines.extend([
                "",
                f"2. FAILURE & RECOVERY SCENARIOS SUMMARY:",
                f"   - Total Failures Injected:      {fm['total_failures_injected']}",
                f"   - Failures Detected:            {fm['failures_detected']} ({fm['detection_rate'] * 100:.0f}%)",
                f"   - Cause Communicated:           {fm['failures_communicated']} ({fm['communication_rate'] * 100:.0f}%)",
                f"   - Alternative Plan Proposed:    {fm['plans_revised']} ({fm['replanning_rate'] * 100:.0f}%)",
                f"   - Post-Failure Recovery:        {fm['failures_recovered']} ({fm['recovery_rate'] * 100:.0f}%)",
            ])

        active_intents = {k: f"{v}%" for k, v in comm_metrics.communication_function_distribution.items() if v > 0}
        lines.extend([
            "",
            f"3. NATURAL-LANGUAGE COMMUNICATION METRICS:",
            f"   - Number of Peer Messages:      {comm_metrics.number_of_messages}",
            f"   - Total Communication Tokens:   {comm_metrics.total_communication_tokens}",
            f"   - Messages per Successful Task: {comm_metrics.messages_per_successful_task if comm_metrics.messages_per_successful_task is not None else 'N/A (Task Incomplete)'}",
            f"   - Tokens per Successful Task:   {comm_metrics.tokens_per_successful_task if comm_metrics.tokens_per_successful_task is not None else 'N/A (Task Incomplete)'}",
            f"   - Avg Message Length:           {comm_metrics.average_message_length_tokens:.1f} tokens ({comm_metrics.average_message_length_chars:.1f} chars)",
            f"   - Grounded Message Rate:        {comm_metrics.grounded_message_rate * 100:.1f}% ({comm_metrics.ungrounded_messages_count} ungrounded)",
            f"   - Redundant Message Rate:       {comm_metrics.redundant_message_rate * 100:.1f}% ({comm_metrics.redundant_messages_count} redundant)",
            f"   - Speech Act Distribution:      {active_intents if active_intents else 'None'}",
            f"   - Comm Volume / Progress Ratio: {comm_metrics.comm_volume_to_progress_ratio} msgs/step",
            "=" * 70,
        ])
        return "\n".join(lines)


__all__ = [
    "FailureType",
    "FailureTrigger",
    "FailureEvent",
    "FailureScenarioSuite",
    "FailureManager",
    "TaskPerformanceMetrics",
    "CommunicationMetrics",
    "MetricsEngine",
]

