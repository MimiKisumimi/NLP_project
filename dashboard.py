"""
Gradio replay dashboard and visualization UI (`dashboard.py`).

Loads stored JSON execution traces (`runs/*.json`) for step-by-step playback (`t = 0 ... T`)
and renders workstation states, robot capability profiles, Task Dependency Graph (DAG) node
cards, LLM Judge bar plots, filtered inter-agent messages, and episode metric scorecards.
"""

import os
import re
import glob
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

from models import CommunicationIntent, LimitationCondition
from tasks import (
    TaskBenchmarkSuite,
    evaluate_dag_node_status,
    evaluate_dag_node_completion,
)
from environment import DEFAULT_WORKSTATIONS
from experiments import BatchExperimentRunner, ExperimentMatrixConfig


# ---------------------------------------------------------
# 1. Trace Discovery, Normalization & On-Demand Generation
# ---------------------------------------------------------
class TraceReplayEngine:
    """
    Loads, normalizes, and indexes stored execution traces (`runs/*.json`) for
    timestep-by-timestep playback ($t = 0 \\dots T$) and generates HTML visualizations
    for the Gradio interface.
    """

    ROBOT_COLORS: Dict[str, str] = {
        "Robot_1": "#2563eb",  # Blue (Carrier / Generalist 1)
        "Robot_2": "#059669",  # Emerald (Manipulation / Generalist 2)
        "Robot_3": "#d97706",  # Amber/Orange (Heat-resistant / Generalist 3)
        "Robot_4": "#7c3aed",  # Violet
    }

    STATION_STYLES: Dict[str, Dict[str, str]] = {
        "Refrigerator": {
            "bg": "#dbeafe",
            "border": "#3b82f6",
            "icon": "🧊",
            "code": "FRG",
        },
        "Cutting table": {
            "bg": "#fef3c7",
            "border": "#f59e0b",
            "icon": "🔪",
            "code": "CUT",
        },
        "Sink": {"bg": "#e0f2fe", "border": "#0284c7", "icon": "🚰", "code": "SNK"},
        "Serving table": {
            "bg": "#dcfce7",
            "border": "#16a34a",
            "icon": "🍽️",
            "code": "SRV",
        },
        "Stove": {"bg": "#fee2e2", "border": "#ef4444", "icon": "🔥", "code": "STV"},
        "Food assembly table": {
            "bg": "#f3e8ff",
            "border": "#9333ea",
            "icon": "🥗",
            "code": "ASM",
        },
    }

    INTENT_BADGE_COLORS: Dict[str, Tuple[str, str]] = {
        "PROPOSAL": ("#dbeafe", "#1e40af"),
        "COMMITMENT": ("#dcfce7", "#166534"),
        "REQUEST": ("#fef3c7", "#92400e"),
        "STATUS_UPDATE": ("#f1f5f9", "#1e293b"),
        "WARNING": ("#ffedd5", "#9a3412"),
        "CONFLICT_RESOLUTION": ("#f3e8ff", "#6b21a8"),
        "PLAN_REVISION": ("#e0e7ff", "#3730a3"),
        "FAILURE_EXPLANATION": ("#fee2e2", "#991b1b"),
    }

    @staticmethod
    def list_trace_files(runs_dir: str = "runs") -> List[str]:
        """Returns sorted list of execution trace JSON files in `runs_dir`."""
        if not os.path.isdir(runs_dir):
            return []
        excluded = {
            "statistical_evaluation_report.json",
            "human_judge_annotation_sample.json",
            "human_judge_agreement_report.json",
        }
        paths = []
        for p in glob.glob(os.path.join(runs_dir, "*.json")):
            if os.path.basename(p) not in excluded:
                paths.append(p)
        paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return paths

    @staticmethod
    def generate_custom_trace(
        task_index: int = 1,
        condition_str: str = "capability_limited",
        repetition: int = 1,
        seed: int = 42,
        runs_dir: str = "runs",
    ) -> str:
        """Runs a single episode on demand and exports its JSON trace for immediate replay."""
        cond = LimitationCondition.from_string(condition_str)
        cfg = ExperimentMatrixConfig(
            task_indices=[task_index],
            conditions=[cond],
            repetitions=repetition,
            base_seed=seed,
            output_dir=runs_dir,
        )
        runner = BatchExperimentRunner(config=cfg, verbose=False)
        rec = runner.run_single_episode(
            task_index=task_index, condition=cond, repetition=repetition
        )
        return rec.trace_filepath

    @staticmethod
    def load_trace(filepath: str) -> Dict[str, Any]:
        """Loads a trace JSON file from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_max_timestep(trace: Dict[str, Any]) -> int:
        """Returns the maximum timestep T in the trace."""
        snaps = trace.get("global_environment_states", [])
        if snaps:
            return max(int(s.get("timestep", 0)) for s in snaps)
        t_traces = trace.get("timestep_traces", [])
        if t_traces:
            return max(int(t.get("timestep", 0)) for t in t_traces)
        return int(trace.get("final_task_state", {}).get("final_timestep", 0))

    @classmethod
    def get_state_at_timestep(
        cls, trace: Dict[str, Any], timestep: int
    ) -> Dict[str, Any]:
        """
        Reconstructs the complete environment snapshot at `timestep` ($t = 0 \\dots T$),
        including robot locations, robot inventories, station items/tools, workstation/tool locks,
        and task DAG progress.
        """
        init_state = trace.get("initial_state", {})
        task_info = trace.get("task", {})

        if timestep <= 0:
            items_dict = init_state.get("items", {})
            ws_items: Dict[str, List[Dict[str, Any]]] = {
                ws.name: [] for ws in DEFAULT_WORKSTATIONS
            }
            robot_invs: Dict[str, List[Dict[str, Any]]] = {
                r_id: [] for r_id in trace.get("agent_profiles", {}).keys()
            }
            for it_id, it_data in items_dict.items():
                loc = it_data.get("location", "")
                if loc in ws_items:
                    ws_items[loc].append(it_data)
                elif loc in robot_invs:
                    robot_invs[loc].append(it_data)

            return {
                "timestep": 0,
                "robot_locations": init_state.get("robot_locations", {}),
                "robot_inventories": robot_invs,
                "items": items_dict,
                "station_items_detailed": ws_items,
                "workstation_locks": {},
                "tool_locks": {},
                "task_progress": init_state.get(
                    "task_progress", task_info.get("progress", {})
                ),
                "agent_observations": {},
            }

        # Search global_environment_states or timestep_traces for matching timestep
        target_snap = None
        for snap in trace.get("global_environment_states", []):
            if int(snap.get("timestep", -1)) == timestep:
                target_snap = snap
                break
        if target_snap is None:
            # Fallback to closest <= timestep
            candidates = [
                s
                for s in trace.get("global_environment_states", [])
                if int(s.get("timestep", 0)) <= timestep
            ]
            if candidates:
                target_snap = candidates[-1]

        obs_at_t = {}
        for tt in trace.get("timestep_traces", []):
            if int(tt.get("timestep", -1)) == timestep:
                obs_at_t = tt.get("agent_observations", {})
                if target_snap is None:
                    target_snap = tt.get("global_state", {})
                break

        if not target_snap:
            target_snap = init_state

        items_dict = target_snap.get("items", {})
        ws_items = {ws.name: [] for ws in DEFAULT_WORKSTATIONS}
        for it_id, it_data in items_dict.items():
            loc = it_data.get("location", "")
            if loc in ws_items:
                ws_items[loc].append(it_data)

        robot_invs = target_snap.get("robot_inventories", {})
        if not robot_invs:
            robot_invs = {r_id: [] for r_id in trace.get("agent_profiles", {}).keys()}
            for it_id, it_data in items_dict.items():
                loc = it_data.get("location", "")
                if loc in robot_invs:
                    robot_invs[loc].append(it_data)

        return {
            "timestep": int(target_snap.get("timestep", timestep)),
            "robot_locations": target_snap.get("robot_locations", {}),
            "robot_inventories": robot_invs,
            "items": items_dict,
            "station_items_detailed": ws_items,
            "workstation_locks": target_snap.get("workstation_locks", {}),
            "tool_locks": target_snap.get("tool_locks", {}),
            "task_progress": target_snap.get(
                "task_progress", task_info.get("progress", {})
            ),
            "agent_observations": obs_at_t,
        }

    @staticmethod
    def resolve_task_dag_nodes(
        trace: Dict[str, Any], state_at_t: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Resolves the Task Dependency Graph (DAG) nodes for the trace's task and annotates
        each node with its dynamic status at `state_at_t`:
        - `completed`: Node is physically completed based on simulator state at timestep t
        - `ready`: Node is not completed, all prerequisites completed, and physical preconditions met
        - `blocked`: Node is not completed, and at least one prerequisite is still incomplete
        """
        task_dict = trace.get("task", {})
        dag_dict = task_dict.get("dependency_graph", {})

        # If trace was exported before dependency_graph was added to Task.to_dict(), reconstruct from TaskBenchmarkSuite
        if not dag_dict:
            task_id = task_dict.get("task_id") or task_dict.get("name", "task_1")
            benchmark_task = TaskBenchmarkSuite.get_task(task_id)
            if benchmark_task is None:
                for t_cand in TaskBenchmarkSuite.get_all_tasks():
                    if t_cand.name.lower() == str(task_dict.get("name", "")).lower():
                        benchmark_task = t_cand
                        break
            if benchmark_task:
                dag_dict = {
                    k: n.to_dict() for k, n in benchmark_task.dependency_graph.items()
                }

        items_dict = state_at_t.get("items", {})
        items_list = (
            list(items_dict.values())
            if isinstance(items_dict, dict)
            else list(items_dict)
        )
        robot_locs = state_at_t.get("robot_locations", {})
        robot_invs = state_at_t.get("robot_inventories", {})

        nodes_out: List[Dict[str, Any]] = []
        completed_cache: Dict[str, bool] = {}

        if items_list:
            for node_id, n_data in dag_dict.items():
                prereqs = list(n_data.get("prerequisites", []))
                status, missing_prereqs = evaluate_dag_node_status(
                    node_id=node_id,
                    node_data=n_data,
                    all_nodes=dag_dict,
                    items=items_list,
                    robot_locations=robot_locs,
                    robot_inventories=robot_invs,
                    completed_cache=completed_cache,
                )
                nodes_out.append(
                    {
                        "node_id": node_id,
                        "description": n_data.get("description", ""),
                        "action_type": n_data.get("action_type", ""),
                        "target_station": n_data.get("target_station") or "Any",
                        "target_item": n_data.get("target_item") or "-",
                        "prerequisites": prereqs,
                        "missing_prerequisites": missing_prereqs,
                        "status": status,
                    }
                )
        else:
            prog = state_at_t.get("task_progress", {}) or {}
            completed_set = set(prog.get("completed_list", []))
            ready_set = set(prog.get("ready_nodes", []))

            for node_id, n_data in dag_dict.items():
                prereqs = list(n_data.get("prerequisites", []))
                if node_id in completed_set:
                    status = "completed"
                elif node_id in ready_set or all(p in completed_set for p in prereqs):
                    status = "ready"
                else:
                    status = "blocked"

                missing_prereqs = [p for p in prereqs if p not in completed_set]
                nodes_out.append(
                    {
                        "node_id": node_id,
                        "description": n_data.get("description", ""),
                        "action_type": n_data.get("action_type", ""),
                        "target_station": n_data.get("target_station") or "Any",
                        "target_item": n_data.get("target_item") or "-",
                        "prerequisites": prereqs,
                        "missing_prerequisites": missing_prereqs,
                        "status": status,
                    }
                )
        return nodes_out

    @staticmethod
    def filter_messages(
        trace: Dict[str, Any],
        max_timestep: Optional[int] = None,
        sender_filter: str = "ALL",
        recipient_filter: str = "ALL",
        intent_filter: str = "ALL",
        include_system: bool = False,
        search_query: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Filters inter-agent messages by timestep, sender, recipient, communication intent,
        system vs. peer status, and free-text search.
        """
        all_msgs = trace.get("agent_messages", [])
        filtered: List[Dict[str, Any]] = []
        q_clean = (search_query or "").strip().lower()

        for m in all_msgs:
            sender = str(m.get("sender", ""))
            is_peer = m.get("is_peer_message", sender != "System")
            if not include_system and not is_peer:
                continue

            t_val = int(m.get("timestep", 0))
            if max_timestep is not None and t_val > max_timestep:
                continue

            if sender_filter and sender_filter != "ALL" and sender != sender_filter:
                continue

            recip = str(m.get("recipient", "ALL"))
            if (
                recipient_filter
                and recipient_filter != "ALL"
                and recip != recipient_filter
            ):
                continue

            intent_str = str(m.get("intent") or m.get("function") or "STATUS_UPDATE")
            if intent_filter and intent_filter != "ALL" and intent_str != intent_filter:
                continue

            text = str(m.get("message_content") or m.get("content") or "")
            if (
                q_clean
                and q_clean not in text.lower()
                and q_clean not in sender.lower()
                and q_clean not in intent_str.lower()
            ):
                continue

            filtered.append(
                {
                    "timestep": t_val,
                    "sender": sender,
                    "recipient": recip,
                    "intent": intent_str,
                    "content": text,
                    "previous_action": str(m.get("previous_action") or "NONE"),
                    "local_location": str(
                        (m.get("local_state") or {}).get("location")
                        or (m.get("local_state") or {}).get("my_location", "-")
                    ),
                    "is_peer": is_peer,
                }
            )
        return filtered

    @staticmethod
    def filter_actions_and_conflicts(
        trace: Dict[str, Any],
        max_timestep: Optional[int] = None,
        robot_filter: str = "ALL",
        status_filter: str = "ALL",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns (filtered_actions, filtered_resource_conflicts) up to `max_timestep`
        with categorization of rejected actions, collisions, and resource lock conflicts.
        """
        actions_out: List[Dict[str, Any]] = []
        for entry in trace.get("actions_and_results", []):
            t_val = int(entry.get("timestep", 0))
            if max_timestep is not None and t_val > max_timestep:
                continue
            r_id = str(entry.get("robot_id", ""))
            if robot_filter and robot_filter != "ALL" and r_id != robot_filter:
                continue

            status = str(entry.get("status", "completed"))
            if status == "completed":
                act_dict = entry.get("action", {}) or {}
                func = str(act_dict.get("function", ""))
                target = (
                    act_dict.get("target_item")
                    or act_dict.get("station")
                    or act_dict.get("target")
                    or "-"
                )
                dur = act_dict.get("duration", 1)
                detail = f"Completed {func} on '{target}' (duration: {dur}s)"
                category = "Valid Execution"
            else:
                func = str(entry.get("function", ""))
                err = str(entry.get("error", "Rejected"))
                detail = err
                err_low = err.lower()
                if "already at" in err_low or "destination" in err_low:
                    category = "Navigation Error"
                elif (
                    "resource conflict" in err_low
                    or "locked" in err_low
                    or "occupied" in err_low
                ):
                    category = "Resource Lock Conflict"
                elif "lacks required skill" in err_low or "not compatible" in err_low:
                    category = "Capability Mismatch"
                elif "carrying capacity" in err_low:
                    category = "Carrying Capacity Limit"
                elif "communication limit" in err_low or "budget" in err_low:
                    category = "Comm Budget Limit"
                elif "injected" in err_low or "slip" in err_low:
                    category = "Injected Hardware/Slip Failure"
                else:
                    category = "Precondition / Dependency Error"

            if status_filter == "COMPLETED_ONLY" and status != "completed":
                continue
            if status_filter == "REJECTED_ONLY" and status == "completed":
                continue
            if status_filter == "NAVIGATION_AND_CONFLICTS" and category not in (
                "Navigation Error",
                "Resource Lock Conflict",
            ):
                continue
            if status_filter == "INJECTED_FAILURES_ONLY" and category != "Injected Hardware/Slip Failure" and "injected" not in detail.lower() and "slip" not in detail.lower() and "misplaced" not in detail.lower():
                continue

            actions_out.append(
                {
                    "timestep": t_val,
                    "robot_id": r_id,
                    "status": (
                        "✅ Completed" if status == "completed" else "❌ Rejected"
                    ),
                    "raw_status": status,
                    "action_type": func,
                    "category": category,
                    "details": detail,
                }
            )

        conflicts_out: List[Dict[str, Any]] = []
        for rc in trace.get("resource_conflicts", []):
            t_val = int(rc.get("timestep", 0))
            if max_timestep is not None and t_val > max_timestep:
                continue
            r_id = str(rc.get("robot_id", ""))
            if (
                robot_filter
                and robot_filter != "ALL"
                and r_id != robot_filter
                and str(rc.get("conflicted_with", "")) != robot_filter
            ):
                continue
            conflicts_out.append(
                {
                    "timestep": t_val,
                    "requesting_robot": r_id,
                    "held_by_robot": str(rc.get("conflicted_with", "")),
                    "resource_type": str(rc.get("resource_type", "")),
                    "resource_name": str(rc.get("resource_name", "")),
                    "attempted_action": str(rc.get("attempted_action", "")),
                }
            )

        return actions_out, conflicts_out

    # ---------------------------------------------------------
    # 2. Rich HTML Component Renderers & Plot Data Builders
    # ---------------------------------------------------------
    FAILURE_TYPE_META: Dict[str, Dict[str, str]] = {
        "tool_unavailable": {
            "name": "Tool Temporarily Unavailable",
            "icon": "🔧",
            "badge_bg": "#fee2e2",
            "badge_fg": "#991b1b",
            "border": "#f87171",
            "bg": "#fef2f2",
            "advice": "A required tool (knife, pan, etc.) is locked/misplaced. Robots must detect failure, announce via FAILURE_EXPLANATION, and switch tasks (e.g. fetch other items from Refrigerator).",
        },
        "workstation_blocked": {
            "name": "Workstation Occupied / Blocked",
            "icon": "🚧",
            "badge_bg": "#ffedd5",
            "badge_fg": "#9a3412",
            "border": "#fb923c",
            "bg": "#fff7ed",
            "advice": "Workstation access is obstructed. Robots should issue WARNING, negotiate priority via CONFLICT_RESOLUTION, or revise plans to another station.",
        },
        "action_execution_slip": {
            "name": "Attempted Action Execution Slip",
            "icon": "⚠️",
            "badge_bg": "#fef3c7",
            "badge_fg": "#92400e",
            "border": "#fcd34d",
            "bg": "#fffbeb",
            "advice": "A valid manipulation action failed unexpectedly due to gripper/mechanical slip. Robots must detect the error, alert peers, and retry or replan.",
        },
        "misplaced_delivery": {
            "name": "Object Delivered to Wrong Location",
            "icon": "📦",
            "badge_bg": "#f3e8ff",
            "badge_fg": "#6b21a8",
            "border": "#c084fc",
            "bg": "#faf5ff",
            "advice": "An item slipped during DROP and was misplaced at the wrong workstation (e.g. Sink/Refrigerator). Robots should locate the misplaced item and coordinate recovery transport via Carrier.",
        },
        "message_delay_or_drop": {
            "name": "Status Message Delayed or Omitted",
            "icon": "📡",
            "badge_bg": "#e0f2fe",
            "badge_fg": "#0369a1",
            "border": "#38bdf8",
            "bg": "#f0f9ff",
            "advice": "An inter-agent message was delayed or dropped, testing decentralized communication resilience and recovery.",
        },
    }

    @classmethod
    def get_failure_injections(cls, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts structured failure injection metadata and events from trace.
        Checks top-level `failure_injections`, then `objective_metrics.task_performance.failure_recovery_summary`,
        and falls back to scanning system messages, action results, and environment locks.
        """
        if not trace:
            return {"configured_triggers": [], "events": [], "total_failures_injected": 0}

        # 1. Direct top-level key
        fi = trace.get("failure_injections")
        if isinstance(fi, dict) and (fi.get("events") or fi.get("configured_triggers")):
            return fi

        # 2. Objective metrics task_performance failure_recovery_summary
        perf = trace.get("objective_metrics", {}).get("task_performance", {})
        if isinstance(perf, dict):
            frs = perf.get("failure_recovery_summary")
            if isinstance(frs, dict) and (frs.get("events") or frs.get("configured_triggers")):
                return frs

        # 3. Dynamic reconstruction from messages, action results, and environment locks
        events = []
        seen_events = set()
        triggers = []

        for m in trace.get("agent_messages", []):
            content = str(m.get("content") or m.get("message_content") or "")
            prev = str(m.get("previous_action", ""))
            t_val = int(m.get("timestep", 0))
            is_fail_msg = (
                prev == "FAILURE_INJECTION"
                or (
                    m.get("sender") == "System"
                    and any(
                        w in content.lower()
                        for w in (
                            "alert:",
                            "delivery error",
                            "injected slip",
                            "network interference",
                            "misplaced at",
                            "temporarily unavailable",
                            "occupied/blocked",
                        )
                    )
                )
            )
            if is_fail_msg:
                ft = "tool_unavailable"
                dur = 4
                res = None
                clow = content.lower()
                if "tool" in clow:
                    ft = "tool_unavailable"
                    m_tool = re.search(r"tool '([^']+)'", content, re.IGNORECASE)
                    res = m_tool.group(1) if m_tool else "tool"
                    m_until = re.search(r"until t=(\d+)s", content)
                    if m_until:
                        dur = max(1, int(m_until.group(1)) - t_val)
                elif "workstation" in clow or "occupied" in clow or "blocked" in clow:
                    ft = "workstation_blocked"
                    m_ws = re.search(r"workstation '([^']+)'", content, re.IGNORECASE)
                    res = m_ws.group(1) if m_ws else "Cutting table"
                    m_until = re.search(r"until t=(\d+)s", content)
                    if m_until:
                        dur = max(1, int(m_until.group(1)) - t_val)
                elif "delivery" in clow or "misplaced" in clow:
                    ft = "misplaced_delivery"
                    dur = 1
                    m_it = re.search(r"item '([^']+)'", content, re.IGNORECASE)
                    res = m_it.group(1) if m_it else "item"
                elif "slip" in clow or "mechanical" in clow:
                    ft = "action_execution_slip"
                    dur = 1
                elif "interference" in clow or "delayed" in clow:
                    ft = "message_delay_or_drop"
                    dur = 3

                ev_id = f"ev_msg_{t_val}_{ft}"
                if ev_id not in seen_events:
                    seen_events.add(ev_id)
                    events.append({
                        "event_id": ev_id,
                        "failure_type": ft,
                        "timestep_injected": t_val,
                        "duration": dur,
                        "description": content,
                        "affected_resource": res,
                        "detected": True,
                        "detected_at_timestep": t_val,
                    })

        for a in trace.get("actions_and_results", []):
            err = str(a.get("error", ""))
            if "injected" in err.lower() or "slip" in err.lower():
                t_val = int(a.get("timestep", 0))
                ev_id = f"ev_act_{t_val}_slip"
                if ev_id not in seen_events:
                    seen_events.add(ev_id)
                    events.append({
                        "event_id": ev_id,
                        "failure_type": "action_execution_slip",
                        "timestep_injected": t_val,
                        "duration": 1,
                        "description": err,
                        "affected_agent": a.get("robot_id"),
                        "affected_resource": a.get("function"),
                        "detected": True,
                        "detected_at_timestep": t_val,
                    })

        # Check workstation & tool locks from global environment states
        for snap in trace.get("global_environment_states", []):
            t_val = int(snap.get("timestep", 0))
            for ws_name, lk in snap.get("workstation_locks", {}).items():
                if isinstance(lk, dict) and lk.get("locked_by") == "FAILURE_INJECTION":
                    ev_id = f"ev_lock_{ws_name}_{lk.get('action')}"
                    if ev_id not in seen_events:
                        seen_events.add(ev_id)
                        until_t = int(lk.get("until_timestep", t_val + 4))
                        events.append({
                            "event_id": ev_id,
                            "failure_type": "workstation_blocked",
                            "timestep_injected": t_val,
                            "duration": max(1, until_t - t_val),
                            "description": f"Workstation '{ws_name}' temporarily blocked/occupied until t={until_t}s.",
                            "affected_resource": ws_name,
                            "detected": True,
                            "detected_at_timestep": t_val,
                        })
            for tid, lk in snap.get("tool_locks", {}).items():
                if isinstance(lk, dict) and lk.get("locked_by") == "FAILURE_INJECTION":
                    tname = lk.get("tool_name", "tool")
                    ev_id = f"ev_lock_tool_{tid}_{tname}"
                    if ev_id not in seen_events:
                        seen_events.add(ev_id)
                        until_t = int(lk.get("until_timestep", t_val + 4))
                        events.append({
                            "event_id": ev_id,
                            "failure_type": "tool_unavailable",
                            "timestep_injected": t_val,
                            "duration": max(1, until_t - t_val),
                            "description": f"Tool '{tname}' ({tid}) temporarily unavailable/locked until t={until_t}s.",
                            "affected_resource": tname,
                            "detected": True,
                            "detected_at_timestep": t_val,
                        })

        return {
            "configured_triggers": triggers,
            "events": events,
            "total_failures_injected": len(events),
            "detection_rate": 1.0 if events else 1.0,
            "recovery_rate": 1.0 if events else 1.0,
        }

    @classmethod
    def render_failure_banner_html(
        cls, trace: Dict[str, Any], timestep: int, state_at_t: Dict[str, Any]
    ) -> str:
        """
        Renders an interactive, timestep-aware Failure Injection Alert Banner
        displayed directly above the workstation grid and DAG playback.
        Indicates:
        - When a failure is currently ACTIVE at timestep `t` (with type, affected resource, and guidance)
        - When an upcoming failure is scheduled
        - When a failure has been successfully resolved
        - Or if this was a clean run with no failures.
        """
        summary = cls.get_failure_injections(trace)
        events = summary.get("events", [])
        triggers = summary.get("configured_triggers", [])

        # Active locks from state snapshot
        active_ws_fail_locks = {
            ws: lk
            for ws, lk in state_at_t.get("workstation_locks", {}).items()
            if isinstance(lk, dict) and lk.get("locked_by") == "FAILURE_INJECTION"
        }
        active_tool_fail_locks = {
            tid: lk
            for tid, lk in state_at_t.get("tool_locks", {}).items()
            if isinstance(lk, dict) and lk.get("locked_by") == "FAILURE_INJECTION"
        }

        # Find active events at `timestep`
        active_events = []
        upcoming_events = []
        resolved_events = []

        for ev in events:
            t_inj = int(ev.get("timestep_injected", 0))
            dur = int(ev.get("duration", 1))
            t_end = t_inj + dur
            ft = str(ev.get("failure_type", "injected_failure")).lower()

            is_active = (t_inj <= timestep < t_end) or (t_inj == timestep)
            if ft == "workstation_blocked" and active_ws_fail_locks:
                is_active = True
            if ft == "tool_unavailable" and active_tool_fail_locks:
                is_active = True

            if is_active:
                active_events.append(ev)
            elif timestep < t_inj:
                upcoming_events.append(ev)
            else:
                resolved_events.append(ev)

        # Also check configured triggers if events haven't fired yet
        for trig in triggers:
            t_inj = int(trig.get("trigger_timestep", 0))
            dur = int(trig.get("duration", 1))
            t_end = t_inj + dur
            ft = str(trig.get("failure_type", "")).lower()
            if ft and ft != "none":
                matched = any(
                    str(e.get("failure_type", "")).lower() == ft for e in events
                )
                if not matched:
                    if t_inj <= timestep < t_end:
                        active_events.append({
                            "event_id": trig.get("trigger_id", "trig"),
                            "failure_type": ft,
                            "timestep_injected": t_inj,
                            "duration": dur,
                            "description": trig.get("description", "Injected perturbation"),
                            "affected_resource": trig.get("target_tool") or trig.get("target_station"),
                        })
                    elif timestep < t_inj:
                        upcoming_events.append({
                            "event_id": trig.get("trigger_id", "trig"),
                            "failure_type": ft,
                            "timestep_injected": t_inj,
                            "duration": dur,
                            "description": trig.get("description", "Scheduled failure"),
                        })
                    else:
                        resolved_events.append({
                            "event_id": trig.get("trigger_id", "trig"),
                            "failure_type": ft,
                            "timestep_injected": t_inj,
                            "duration": dur,
                            "description": trig.get("description", "Resolved failure"),
                        })

        # Case 1: Active Injected Failure(s) at this second!
        if active_events:
            event_items = []
            for ev in active_events:
                ft_key = str(ev.get("failure_type", "injected_failure")).lower()
                meta = cls.FAILURE_TYPE_META.get(
                    ft_key,
                    {
                        "name": "Controlled Failure Injection",
                        "icon": "💥",
                        "badge_bg": "#fee2e2",
                        "badge_fg": "#991b1b",
                        "border": "#f87171",
                        "bg": "#fef2f2",
                        "advice": "Adapt plan and coordinate recovery via peer messaging.",
                    },
                )
                t_inj = ev.get("timestep_injected", 0)
                dur = ev.get("duration", 1)
                t_end = t_inj + dur
                rem = max(0, t_end - timestep)
                desc = ev.get("description", "")
                res = ev.get("affected_resource")
                res_badge = (
                    f'<span style="background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; font-weight:700; padding:2px 8px; border-radius:4px; font-size:11px;">Target: <b>{res}</b></span>'
                    if res
                    else ""
                )
                dur_badge = (
                    f'<span style="background:#fee2e2; color:#b91c1c; font-weight:700; padding:2px 8px; border-radius:4px; font-size:11px; border:1px solid #fca5a5;">⏳ Active for {rem}s more (t={t_inj}s–{t_end}s)</span>'
                    if dur > 1
                    else f'<span style="background:#fee2e2; color:#b91c1c; font-weight:700; padding:2px 8px; border-radius:4px; font-size:11px; border:1px solid #fca5a5;">💥 Injected at t={t_inj}s</span>'
                )

                event_items.append(
                    f'<div style="background:{meta["bg"]}; border:2px solid {meta["border"]}; border-radius:8px; padding:10px 14px; margin-top:8px;">'
                    f'  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                    f'    <div style="display:flex; align-items:center; gap:8px;">'
                    f'      <span style="font-size:18px;">{meta["icon"]}</span>'
                    f'      <span style="font-size:14px; font-weight:800; color:{meta["badge_fg"]};">{meta["name"]}</span>'
                    f'      {res_badge}'
                    f"    </div>"
                    f"    <div>{dur_badge}</div>"
                    f"  </div>"
                    f'  <div style="font-size:12.5px; color:#1e293b; font-weight:600; margin-top:6px; line-height:1.4;">{desc}</div>'
                    f'  <div style="font-size:11.5px; color:#475569; margin-top:5px; background:rgba(255,255,255,0.7); border-radius:6px; padding:5px 8px;">'
                    f'    <b style="color:#0f172a;">Expected Resilient Behavior:</b> {meta["advice"]}'
                    f"  </div>"
                    f"</div>"
                )

            return (
                f'<div style="font-family: system-ui, -apple-system, sans-serif; background:#fff1f2; border:2px solid #ef4444; border-radius:10px; padding:12px 16px; margin-bottom:12px; box-shadow:0 2px 6px rgba(239,68,68,0.15);">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center;">'
                f'    <div style="font-size:14px; font-weight:800; color:#b91c1c; display:flex; align-items:center; gap:8px;">'
                f'      <span style="font-size:18px;">🚨</span>'
                f'      <span>FAILURE INJECTION ACTIVE AT TIMESTEP t = {timestep}s</span>'
                f"    </div>"
                f'    <span style="background:#b91c1c; color:#ffffff; font-size:11px; font-weight:700; padding:3px 10px; border-radius:999px;">{len(active_events)} ACTIVE PERTURBATION(S)</span>'
                f"  </div>"
                f'  {"".join(event_items)}'
                f"</div>"
            )

        # Case 2: Upcoming or recently resolved failures in this run
        if upcoming_events or resolved_events:
            upcoming_badges = []
            for ev in upcoming_events:
                ft_key = str(ev.get("failure_type", "")).lower()
                meta = cls.FAILURE_TYPE_META.get(ft_key, {"name": "Injected Failure", "icon": "⚠️"})
                t_inj = ev.get("timestep_injected", 0)
                upcoming_badges.append(
                    f'<span style="background:#fffbeb; color:#92400e; border:1px solid #fde68a; padding:3px 8px; border-radius:6px; font-size:11px; font-weight:600;">'
                    f'⏳ {meta["icon"]} {meta["name"]} scheduled at t={t_inj}s (in {t_inj - timestep}s)</span>'
                )

            resolved_badges = []
            for ev in resolved_events:
                ft_key = str(ev.get("failure_type", "")).lower()
                meta = cls.FAILURE_TYPE_META.get(ft_key, {"name": "Injected Failure", "icon": "✅"})
                t_inj = ev.get("timestep_injected", 0)
                dur = ev.get("duration", 1)
                resolved_badges.append(
                    f'<span style="background:#f0fdf4; color:#166534; border:1px solid #bbf7d0; padding:3px 8px; border-radius:6px; font-size:11px; font-weight:600;">'
                    f'✅ {meta["name"]} resolved (was active t={t_inj}s–{t_inj+dur}s)</span>'
                )

            upcoming_html = " ".join(upcoming_badges)
            resolved_html = " ".join(resolved_badges)

            return (
                f'<div style="font-family: system-ui, -apple-system, sans-serif; background:#f8fafc; border:1px solid #cbd5e1; border-left:4px solid #3b82f6; border-radius:10px; padding:10px 14px; margin-bottom:12px;">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                f'    <div style="font-size:13px; font-weight:700; color:#0f172a; display:flex; align-items:center; gap:6px;">'
                f'      <span>ℹ️</span> <b>Controlled Failure Injection Status at t = {timestep}s:</b> No failure active at this second'
                f"    </div>"
                f'    <div style="display:flex; flex-wrap:wrap; gap:6px;">{upcoming_html} {resolved_html}</div>'
                f"  </div>"
                f"</div>"
            )

        # Case 3: Clean run
        return (
            f'<div style="font-family: system-ui, -apple-system, sans-serif; background:#f8fafc; border:1px solid #cbd5e1; border-left:4px solid #10b981; border-radius:10px; padding:10px 14px; margin-bottom:12px;">'
            f'  <div style="display:flex; align-items:center; gap:8px; font-size:12.5px; color:#0f172a;">'
            f'    <span>🛡️</span>'
            f'    <b>Clean Run (No Injected Failures):</b> <span style="color:#475569;">This episode ran without any hardware, tool, workstation, slip, or message delay perturbations.</span>'
            f"  </div>"
            f"</div>"
        )

    @classmethod
    def render_grid_html(cls, trace: Dict[str, Any], state_at_t: Dict[str, Any]) -> str:
        """
        Renders the state-based Kitchen Workstation View at timestep `t` showing:
        - All 6 workstations with color coding, icons, current inventory items, and active locks
        - The robots (`Robot_1`, `Robot_2`, `Robot_3`) currently located at each workstation,
          with role badges and carried items
        """
        t_val = state_at_t.get("timestep", 0)
        robot_locs: Dict[str, Any] = state_at_t.get("robot_locations", {})
        robot_invs: Dict[str, List[Dict[str, Any]]] = state_at_t.get(
            "robot_inventories", {}
        )
        ws_items: Dict[str, List[Dict[str, Any]]] = state_at_t.get(
            "station_items_detailed", {}
        )
        ws_locks: Dict[str, Any] = state_at_t.get("workstation_locks", {})
        tool_locks: Dict[str, Any] = state_at_t.get("tool_locks", {})
        profiles: Dict[str, Any] = trace.get("agent_profiles", {})

        # Map station name -> list of robots located there
        station_to_robots: Dict[str, List[str]] = {}
        for r_id, loc in robot_locs.items():
            station_to_robots.setdefault(str(loc), []).append(r_id)

        html = [
            '<div style="font-family: system-ui, -apple-system, sans-serif; background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">',
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">',
            f'  <div style="font-weight:700; font-size:16px; color:#0f172a;">🍳 Kitchen Workstations — Timestep t = {t_val}s</div>',
            f'  <div style="font-size:12px; color:#475569;">Robot locations, station items &amp; locks</div>',
            f"</div>",
            '<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">',
        ]

        for ws in DEFAULT_WORKSTATIONS:
            st = cls.STATION_STYLES.get(
                ws.name,
                {"bg": "#f8fafc", "border": "#94a3b8", "icon": "📦", "code": ws.code},
            )

            # Station items (ingredients & tools)
            formatted_items = []
            for it in ws_items.get(ws.name, []):
                if it.get("is_tool"):
                    t_lock = tool_locks.get(it.get("id", ""))
                    if isinstance(t_lock, dict) and t_lock.get("locked_by") == "FAILURE_INJECTION":
                        formatted_items.append(
                            f"<span style='background:#fee2e2; color:#991b1b; border:2px solid #ef4444; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:800; box-shadow:0 1px 3px rgba(239,68,68,0.25);'>💥 🔧 {it.get('name')} 🔒[FAILURE INJECTION]</span>"
                        )
                    else:
                        lk_str = (
                            f" 🔒({t_lock.get('locked_by')})"
                            if isinstance(t_lock, dict)
                            else ""
                        )
                        formatted_items.append(
                            f"<span style='background:#f1f5f9; color:#0f172a; border:1px solid #94a3b8; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:600;'>🔧 {it.get('name')}{lk_str}</span>"
                        )
                else:
                    formatted_items.append(
                        f"<span style='background:#ffffff; color:#0f172a; border:1px solid #94a3b8; padding:2px 6px; border-radius:4px; font-size:11px;'><span style='color:#0f172a; font-weight:600;'>🥗 {it.get('name')}</span> <b style='color:#334155;'>({it.get('state')})</b></span>"
                    )
            items_html = (
                " ".join(formatted_items)
                if formatted_items
                else "<span style='color:#64748b; font-size:11px; font-style:italic;'>(empty)</span>"
            )

            # Workstation lock badge
            ws_lk = ws_locks.get(ws.name)
            ws_lk_html = ""
            if ws_lk:
                by = (
                    ws_lk.get("locked_by", "")
                    if isinstance(ws_lk, dict)
                    else str(ws_lk)
                )
                act = ws_lk.get("action", "") if isinstance(ws_lk, dict) else ""
                until = (
                    ws_lk.get("until_timestep", "") if isinstance(ws_lk, dict) else ""
                )
                if by == "FAILURE_INJECTION":
                    ws_lk_html = f'<div style="color:#7f1d1d; background:#fee2e2; border:2px solid #b91c1c; font-size:11px; font-weight:800; margin-top:4px; padding:3px 7px; border-radius:4px; box-shadow:0 1px 4px rgba(185,28,28,0.3);">💥 INJECTED FAILURE: Station Blocked ({act} until t={until}s)</div>'
                else:
                    ws_lk_html = f'<div style="color:#b91c1c; background:#fee2e2; border:1px solid #fca5a5; font-size:11px; font-weight:700; margin-top:4px; padding:2px 6px; border-radius:4px;">🔒 Locked by {by} ({act} until t={until}s)</div>'

            # Robots currently at this station
            robot_badges = []
            for r_id in station_to_robots.get(ws.name, []):
                r_color = cls.ROBOT_COLORS.get(r_id, "#2563eb")
                role = profiles.get(r_id, {}).get("role", "Robot")
                inv = robot_invs.get(r_id, [])
                inv_str = (
                    ", ".join(f"{it.get('name')}({it.get('state')})" for it in inv)
                    if inv
                    else "Empty"
                )
                robot_badges.append(
                    f'<div style="background:#ffffff; border:2px solid {r_color}; border-radius:6px; padding:5px 7px; margin-top:5px; box-shadow:0 1px 2px rgba(0,0,0,0.05); color:#0f172a;">'
                    f'  <div style="display:flex; align-items:center; gap:6px;">'
                    f'    <span style="background:{r_color}; color:#ffffff; font-weight:700; font-size:11px; padding:2px 6px; border-radius:4px;">🤖 {r_id}</span>'
                    f'    <span style="font-size:11px; color:#0f172a; font-weight:700;">{role}</span>'
                    f"  </div>"
                    f'  <div style="font-size:10.5px; color:#1e293b; margin-top:3px;"><b style="color:#0f172a;">🎒 Holding:</b> {inv_str}</div>'
                    f"</div>"
                )
            robots_html = (
                "".join(robot_badges)
                if robot_badges
                else "<div style='color:#64748b; font-size:11px; margin-top:4px; font-style:italic;'>No robot here</div>"
            )

            html.append(
                f'<div style="background:{st["bg"]}; border:2px solid {st["border"]}; border-radius:8px; padding:10px; display:flex; flex-direction:column; color:#0f172a;">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                f'    <span style="font-weight:700; font-size:13px; color:#0f172a;">{st["icon"]} {ws.name}</span>'
                f'    <span style="font-size:10.5px; font-weight:700; color:#475569; background:rgba(255,255,255,0.7); padding:1px 5px; border-radius:4px;">{ws.code}</span>'
                f"  </div>"
                f'  <div style="display:flex; flex-wrap:wrap; gap:4px;">{items_html}</div>'
                f"  {ws_lk_html}"
                f"  {robots_html}"
                f"</div>"
            )

        html.append("</div></div>")
        return "\n".join(html)

    @classmethod
    def render_robot_profiles_html(
        cls, trace: Dict[str, Any], state_at_t: Dict[str, Any]
    ) -> str:
        """
        Renders the Robot Capability Profiles Display Cards
        along with each robot's live state at timestep `t`.
        """
        profiles = trace.get("agent_profiles", {})
        cond_cfg = trace.get("condition_config", {})
        robot_locs = state_at_t.get("robot_locations", {})
        robot_invs = state_at_t.get("robot_inventories", {})
        obs_at_t = state_at_t.get("agent_observations", {})

        obs_mode = (
            "PARTIAL (Sensor Range = " + str(cond_cfg.get("sensor_range", 2)) + ")"
            if cond_cfg.get("partial_observability")
            else "FULL (Global Station Visibility)"
        )

        cards = [
            '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; font-family: system-ui, -apple-system, sans-serif; color:#0f172a;">'
        ]
        for r_id, prof in profiles.items():
            color = cls.ROBOT_COLORS.get(r_id, "#2563eb")
            role = prof.get("role", "Generalist")
            desc = prof.get("description", "")
            skills = prof.get("skills", [])
            cap = prof.get("carrying_capacity", 1)
            tools = prof.get("tool_compatibility", [])
            max_msgs = prof.get("max_messages")
            max_toks = prof.get("max_comm_tokens")
            model_name = prof.get(
                "model", trace.get("acting_agent_model", "qwen2.5:32b")
            )

            loc = str(robot_locs.get(r_id, "-"))
            inv = robot_invs.get(r_id, [])
            inv_str = (
                ", ".join(f"{it.get('name')} ({it.get('state')})" for it in inv)
                if inv
                else "Empty"
            )
            r_obs = obs_at_t.get(r_id, {})
            cur_act = r_obs.get("current_action") or "IDLE / READY"
            busy_rem = r_obs.get("busy_remaining_seconds", 0)
            status_str = (
                f"{cur_act} ({busy_rem}s left)"
                if busy_rem and busy_rem > 0
                else "Ready"
            )

            is_busy = bool(busy_rem and busy_rem > 0)
            if is_busy:
                status_badge = f'<span style="background:#fef3c7; color:#92400e; font-weight:700; padding:2px 8px; border-radius:4px; border:1px solid #fde68a; font-size:11px;">⏳ {status_str}</span>'
            else:
                status_badge = f'<span style="background:#dcfce7; color:#166534; font-weight:700; padding:2px 8px; border-radius:4px; border:1px solid #bbf7d0; font-size:11px;">✅ {status_str}</span>'

            if inv:
                inv_badge = f'<span style="background:#eff6ff; color:#1e40af; font-weight:600; padding:2px 7px; border-radius:4px; border:1px solid #bfdbfe; font-size:11px;">{inv_str}</span>'
            else:
                inv_badge = '<span style="color:#64748b; font-size:11px; font-style:italic;">Empty</span>'

            skills_badges = " ".join(
                f'<span style="background:#dbeafe; color:#1e40af; font-size:10.5px; font-weight:600; padding:2px 7px; border-radius:4px; border:1px solid #bfdbfe;">{s}</span>'
                for s in skills
            )
            tools_badges = (
                " ".join(
                    f'<span style="background:#fef3c7; color:#92400e; font-size:10.5px; font-weight:600; padding:2px 7px; border-radius:4px; border:1px solid #fde68a;">🔧 {t}</span>'
                    for t in tools
                )
                if tools
                else '<span style="color:#64748b; font-size:11px; font-style:italic;">None (No tool use)</span>'
            )

            comm_budget_str = (
                f"{max_msgs} msgs / {max_toks} tokens"
                if (max_msgs is not None or max_toks is not None)
                else "Unlimited"
            )

            cards.append(
                f'<div style="background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; border-top:4px solid {color}; border-radius:10px; padding:14px; box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                f'    <span style="font-size:15px; font-weight:700; color:{color};">🤖 {r_id} — {role}</span>'
                f'    <span style="font-size:11px; background:#f1f5f9; color:#0f172a; border:1px solid #cbd5e1; padding:2px 8px; border-radius:6px; font-weight:600;">📍 {loc}</span>'
                f"  </div>"
                f'  <div style="font-size:11.5px; color:#334155; margin-bottom:10px; line-height:1.4;">{desc}</div>'
                f'  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 10px; margin-bottom:8px; display:flex; flex-wrap:wrap; align-items:center; gap:8px;">'
                f'    <div style="font-size:11.5px; color:#0f172a;"><b style="color:#0f172a;">Live Status (t={state_at_t.get("timestep", 0)}s):</b> {status_badge}</div>'
                f'    <div style="font-size:11.5px; color:#0f172a;"><b style="color:#0f172a;">Holding ({len(inv)}/{cap}):</b> {inv_badge}</div>'
                f"  </div>"
                f'  <div style="font-size:11.5px; color:#0f172a; margin-bottom:8px;">'
                f'    <b style="color:#0f172a;">Allowed Skills:</b>'
                f'    <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">{skills_badges}</div>'
                f"  </div>"
                f'  <div style="font-size:11.5px; color:#0f172a; margin-bottom:8px;">'
                f'    <b style="color:#0f172a;">Tool Compatibility:</b>'
                f'    <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">{tools_badges}</div>'
                f"  </div>"
                f'  <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:11px; color:#1e293b; background:#f8fafc; border:1px solid #e2e8f0; padding:8px 10px; border-radius:6px; margin-top:8px;">'
                f'    <div><b style="color:#0f172a;">Carrying Cap:</b> <span style="color:#1e293b;">{cap} item(s)</span></div>'
                f'    <div><b style="color:#0f172a;">LLM Model:</b> <code style="background:#e2e8f0; color:#0f172a; padding:1px 4px; border-radius:3px; font-weight:600;">{model_name}</code></div>'
                f'    <div><b style="color:#0f172a;">Sensor Mode:</b> <span style="color:#1e293b;">{obs_mode}</span></div>'
                f'    <div><b style="color:#0f172a;">Comm Budget:</b> <span style="color:#1e293b;">{comm_budget_str}</span></div>'
                f"  </div>"
                f"</div>"
            )
        cards.append("</div>")
        return "\n".join(cards)

    @classmethod
    def render_dag_html(cls, trace: Dict[str, Any], state_at_t: Dict[str, Any]) -> str:
        """
        Renders the Dynamic Task Dependency Graph (DAG) View at timestep `t`,
        highlighting Completed (green), Ready / Unblocked (amber), and Blocked (slate/red) nodes
        along with prerequisite edges.
        """
        nodes = cls.resolve_task_dag_nodes(trace, state_at_t)
        total = len(nodes)
        completed_cnt = sum(1 for n in nodes if n["status"] == "completed")
        ready_cnt = sum(1 for n in nodes if n["status"] == "ready")
        blocked_cnt = sum(1 for n in nodes if n["status"] == "blocked")

        status_styles = {
            "completed": ("#dcfce7", "#16a34a", "#166534", "✅ COMPLETED"),
            "ready": ("#fef3c7", "#d97706", "#92400e", "⚡ READY (UNBLOCKED)"),
            "blocked": ("#f1f5f9", "#94a3b8", "#334155", "🔒 BLOCKED"),
        }

        html = [
            '<div style="font-family: system-ui, -apple-system, sans-serif; background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; border-radius:12px; padding:16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">',
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">',
            f'  <div style="font-weight:700; font-size:15px; color:#0f172a;">🕸️ Dynamic Task Dependency Graph — {completed_cnt}/{total} Nodes Completed</div>',
            f'  <div style="display:flex; gap:10px; font-size:12px; font-weight:700;">',
            f'    <span style="color:#15803d;">✅ Completed: {completed_cnt}</span>',
            f'    <span style="color:#b45309;">⚡ Ready: {ready_cnt}</span>',
            f'    <span style="color:#475569;">🔒 Blocked: {blocked_cnt}</span>',
            f"  </div>",
            f"</div>",
            '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px;">',
        ]

        for n in nodes:
            bg, border, fg, badge_txt = status_styles.get(
                n["status"], status_styles["blocked"]
            )
            prereq_str = (
                ", ".join(n["prerequisites"])
                if n["prerequisites"]
                else "None (Root Task)"
            )
            missing_str = ""
            if n["status"] == "blocked" and n["missing_prerequisites"]:
                missing_str = f'<div style="font-size:10px; color:#b91c1c; font-weight:700; margin-top:4px;">⏳ Waiting on: {", ".join(n["missing_prerequisites"])}</div>'

            html.append(
                f'<div style="background:{bg}; border:2px solid {border}; border-radius:8px; padding:10px; color:#0f172a;">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                f'    <span style="font-weight:700; font-size:12px; color:#0f172a;">{n["node_id"]}</span>'
                f'    <span style="font-size:10px; font-weight:700; color:{fg}; background:rgba(255,255,255,0.9); border:1px solid {border}; padding:2px 6px; border-radius:4px;">{badge_txt}</span>'
                f"  </div>"
                f'  <div style="font-size:11px; color:#0f172a; margin-bottom:6px; font-weight:500;">{n["description"]}</div>'
                f'  <div style="font-size:10px; color:#1e293b;"><b style="color:#0f172a;">Action:</b> {n["action_type"]} | <b style="color:#0f172a;">Station:</b> {n["target_station"]} | <b style="color:#0f172a;">Item:</b> {n["target_item"]}</div>'
                f'  <div style="font-size:10px; color:#334155; margin-top:3px;"><b style="color:#0f172a;">Prerequisites:</b> [{prereq_str}]</div>'
                f"  {missing_str}"
                f"</div>"
            )

        html.append("</div></div>")
        return "\n".join(html)

    JUDGE_DIMENSIONS: List[Tuple[str, str]] = [
        ("relevance", "Relevance"),
        ("clarity", "Clarity"),
        ("grounding", "Grounding"),
        ("coordination_utility", "Coord. Utility"),
        ("redundancy", "Non-Redundancy"),
        ("planning_coherence", "Plan Coherence"),
        ("failure_explanation_quality", "Failure Expl."),
    ]

    @classmethod
    def build_judge_scores_dataframe(cls, judge_report: Optional[Dict[str, Any]]):
        """
        Builds a `pandas.DataFrame` with columns `Dimension` and `Score` (1–5 scale) for the 7
        Independent LLM Judge communication dimensions, for display in a standard Gradio `gr.BarPlot`.
        Returns an empty DataFrame when no judge report is available.
        """
        import pandas as pd

        if not judge_report:
            return pd.DataFrame({"Dimension": [], "Score": []})

        scores_map = judge_report.get("scores", {})
        dims_detail = judge_report.get("dimensions", {})
        rows = [
            {
                "Dimension": label,
                "Score": float(
                    scores_map.get(key, dims_detail.get(key, {}).get("score", 3))
                ),
            }
            for key, label in cls.JUDGE_DIMENSIONS
        ]
        return pd.DataFrame(rows)

    @classmethod
    def render_messages_html(cls, messages: List[Dict[str, Any]]) -> str:
        """Renders filtered inter-agent messages with speech-act intent badges and local state context."""
        if not messages:
            return '<div style="padding:14px; color:#334155; background:#f8fafc; border-radius:8px; border:1px solid #cbd5e1; font-family:system-ui, sans-serif;">No messages match the current filters.</div>'

        rows = [
            '<div style="display:flex; flex-direction:column; gap:8px; max-height:460px; overflow-y:auto; font-family:system-ui, -apple-system, sans-serif; padding-right:4px;">'
        ]
        for m in messages:
            sender = m["sender"]
            s_color = cls.ROBOT_COLORS.get(sender, "#475569")
            intent = m["intent"]
            bg_badge, fg_badge = cls.INTENT_BADGE_COLORS.get(
                intent, ("#f1f5f9", "#1e293b")
            )
            is_failure_alert = (
                sender == "System"
                and any(
                    kw in str(m["content"]).lower()
                    for kw in (
                        "alert:",
                        "failure",
                        "interference",
                        "misplaced",
                        "slip",
                        "temporarily unavailable",
                        "occupied/blocked",
                    )
                )
            )
            if is_failure_alert:
                s_color = "#dc2626"
                bg_badge = "#fee2e2"
                fg_badge = "#991b1b"
                intent = "🚨 FAILURE INJECTION ALERT"
                card_style = "background:#fff1f2; color:#0f172a; border:2px solid #ef4444; border-left:6px solid #dc2626; border-radius:8px; padding:10px 12px; box-shadow:0 1px 4px rgba(239,68,68,0.15);"
            else:
                card_style = f"background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; border-left:4px solid {s_color}; border-radius:8px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,0.04);"

            rows.append(
                f'<div style="{card_style}">'
                f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                f'    <div style="display:flex; align-items:center; gap:8px;">'
                f'      <span style="font-weight:700; font-size:12px; color:{s_color};">{sender}</span>'
                f'      <span style="font-size:11px; color:#475569;">➔ <b style="color:#0f172a;">{m["recipient"]}</b></span>'
                f'      <span style="background:{bg_badge}; color:{fg_badge}; border:1px solid {fg_badge}33; font-size:10px; font-weight:700; padding:2px 6px; border-radius:4px;">{intent}</span>'
                f"    </div>"
                f'    <div style="font-size:11px; color:#475569;"><span style="font-weight:600; color:#1e293b;">t = {m["timestep"]}s</span> | Pos: <b style="color:#1e293b;">{m["local_location"]}</b> | Prev: <code style="background:#f1f5f9; color:#0f172a; padding:1px 4px; border-radius:3px; border:1px solid #e2e8f0;">{m["previous_action"]}</code></div>'
                f"  </div>"
                f'  <div style="font-size:12.5px; color:#0f172a; line-height:1.45; margin-top:4px;">{m["content"]}</div>'
                f"</div>"
            )
        rows.append("</div>")
        return "\n".join(rows)

    @classmethod
    def render_metrics_cards_html(cls, trace: Dict[str, Any]) -> str:
        """
        Renders rich visual Metric Cards for Objective Task-Performance,
        Coordination Efficiency, and Natural-Language Communication metrics.
        """
        if not trace:
            return '<div style="padding:14px; color:#64748b; font-family:system-ui, sans-serif;">No trace data loaded.</div>'

        max_t = cls.get_max_timestep(trace)
        obj_metrics = trace.get("objective_metrics") or {}
        perf = obj_metrics.get("task_performance") or {}
        comm = obj_metrics.get("communication") or {}
        judge_rep = trace.get("llm_judge_report") or {}
        judge_scores = (
            (judge_rep.get("scores") or {}) if isinstance(judge_rep, dict) else {}
        )

        task_name = trace.get("task", {}).get("name") or "Unknown Task"
        condition_name = trace.get("condition_config", {}).get("name") or "Reference"
        run_id = trace.get("run_id") or "trace"

        # Task outcome
        task_success = perf.get("task_success")
        if task_success is None:
            final_state = trace.get("final_task_state", {})
            task_success = (
                final_state.get("verification", {}).get("is_complete")
                or final_state.get("is_complete", False)
            )

        status_bg = "#ecfdf5" if task_success else "#fff1f2"
        status_border = "#10b981" if task_success else "#f43f5e"
        status_fg = "#047857" if task_success else "#be123c"
        status_icon = "✅" if task_success else "❌"
        status_label = "TASK SUCCESSFUL" if task_success else "INCOMPLETE / FAILED"
        status_sub = (
            "Ground-truth verifier passed"
            if task_success
            else "Recipe requirements unmet"
        )

        makespan = perf.get("makespan")
        if makespan is None:
            makespan = max_t

        total_actions = perf.get("total_actions", 0)
        completed_actions = perf.get("completed_actions_count", 0)
        failed_actions = perf.get("failed_actions_count", 0)
        if total_actions == 0:
            actions_list = trace.get("actions_and_results", [])
            total_actions = len(actions_list)
            completed_actions = sum(
                1 for a in actions_list if a.get("status") == "completed"
            )
            failed_actions = total_actions - completed_actions

        valid_rate = perf.get("valid_action_rate")
        if valid_rate is None:
            valid_rate = (
                (completed_actions / total_actions * 100.0)
                if total_actions > 0
                else 100.0
            )
        else:
            valid_rate = valid_rate * 100.0 if valid_rate <= 1.0 else valid_rate

        overall_judge = comm.get("llm_judge_overall_score")
        if overall_judge is None and isinstance(judge_rep, dict):
            overall_judge = judge_rep.get("overall_average_score")
        if overall_judge is not None:
            judge_str = f"⭐ {float(overall_judge):.2f} / 5.0"
        else:
            judge_str = "N/A"
        judge_model = (
            judge_rep.get("judge_model", "gemini-3.1-pro-preview")
            if isinstance(judge_rep, dict)
            else "gemini-3.1-pro-preview"
        )

        conflicts = perf.get(
            "resource_conflicts", len(trace.get("resource_conflicts", []))
        )
        dep_errors = perf.get("dependency_errors", 0)
        workload_shares = perf.get("workload_shares", {})
        total_moves = perf.get(
            "total_movements",
            sum(
                1
                for a in trace.get("actions_and_results", [])
                if str(
                    a.get("action", {}).get("function") or a.get("function", "")
                ).upper()
                == "MOVE"
            ),
        )

        num_msgs = comm.get("number_of_messages")
        if num_msgs is None:
            num_msgs = len(trace.get("agent_messages", []))
        total_tokens = comm.get("total_communication_tokens", 0)
        avg_tokens = comm.get("average_message_length_tokens", 0.0)
        if total_tokens == 0 and num_msgs > 0:
            total_tokens = sum(
                len(str(m.get("content") or m.get("message_content", "")).split())
                for m in trace.get("agent_messages", [])
            )
            avg_tokens = round(total_tokens / num_msgs, 1)

        grounded_rate = comm.get("grounded_message_rate", 1.0)
        grounded_pct = (
            (grounded_rate * 100.0) if grounded_rate <= 1.0 else grounded_rate
        )
        ungrounded_cnt = comm.get("ungrounded_messages_count", 0)

        redundant_rate = comm.get("redundant_message_rate", 0.0)
        non_redundant_pct = (
            ((1.0 - redundant_rate) * 100.0)
            if redundant_rate <= 1.0
            else (100.0 - redundant_rate)
        )
        redundant_cnt = comm.get("redundant_messages_count", 0)

        workload_badges = []
        for r_id, share in workload_shares.items():
            r_col = cls.ROBOT_COLORS.get(r_id, "#2563eb")
            pct_val = share * 100.0 if share <= 1.0 else share
            workload_badges.append(
                f'<span style="background:#ffffff; border:1px solid {r_col}; color:#0f172a; padding:2px 7px; border-radius:4px; font-size:11px; font-weight:600;">'
                f'<b style="color:{r_col};">{r_id}:</b> {pct_val:.0f}%</span>'
            )
        workload_html = (
            " ".join(workload_badges)
            if workload_badges
            else '<span style="color:#64748b; font-size:11px; font-style:italic;">Equally distributed</span>'
        )

        intent_dist = comm.get("communication_function_distribution", {})
        intent_badges = []
        for intent_name, pct_val in sorted(
            intent_dist.items(), key=lambda x: x[1], reverse=True
        ):
            if pct_val > 0:
                bg_badge, fg_badge = cls.INTENT_BADGE_COLORS.get(
                    intent_name, ("#f1f5f9", "#1e293b")
                )
                intent_badges.append(
                    f'<span style="background:{bg_badge}; color:{fg_badge}; border:1px solid {fg_badge}33; padding:2px 7px; border-radius:4px; font-size:10.5px; font-weight:700;">'
                    f"{intent_name}: {pct_val:.1f}%</span>"
                )
        intents_html = (
            " ".join(intent_badges)
            if intent_badges
            else '<span style="color:#64748b; font-size:11px; font-style:italic;">No speech acts recorded</span>'
        )

        omitted_perf_keys = {"workload_gini", "dag_progress_pct"}
        omitted_comm_keys = {
            "average_message_length_chars",
            "comm_volume_to_progress_ratio",
        }
        filtered_perf = {
            k: v for k, v in perf.items() if k not in omitted_perf_keys
        }
        filtered_comm = {
            k: v for k, v in comm.items() if k not in omitted_comm_keys
        }

        raw_dict = {
            "run_id": run_id,
            "task": task_name,
            "condition": condition_name,
            "max_timestep": max_t,
            "task_performance_metrics": filtered_perf,
            "communication_metrics": filtered_comm,
            "llm_judge_scores": judge_scores,
        }
        raw_json_str = json.dumps(raw_dict, indent=2)

        html = [
            '<div style="font-family: system-ui, -apple-system, sans-serif; display: flex; flex-direction: column; gap: 12px; color: #0f172a;">',
            "  <!-- Hero Metric Cards Row -->",
            '  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;">',
            f'    <div style="background:{status_bg}; border:2px solid {status_border}; border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
            f'      <div style="font-size:11px; font-weight:700; color:{status_fg}; text-transform:uppercase; letter-spacing:0.5px;">Outcome</div>',
            f'      <div style="font-size:17px; font-weight:800; color:{status_fg}; margin:4px 0;">{status_icon} {status_label}</div>',
            f'      <div style="font-size:11px; color:#475569;">{status_sub}</div>',
            "    </div>",
            '    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
            '      <div style="font-size:11px; font-weight:700; color:#0369a1; text-transform:uppercase; letter-spacing:0.5px;">Makespan</div>',
            f'      <div style="font-size:20px; font-weight:800; color:#0c4a6e; margin:4px 0;">⏱️ {makespan}s</div>',
            f'      <div style="font-size:11px; color:#475569;">Episode duration (limit: {max_t}s)</div>',
            "    </div>",
            '    <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
            '      <div style="font-size:11px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.5px;">Action Validity</div>',
            f'      <div style="font-size:20px; font-weight:800; color:#0f172a; margin:4px 0;">⚡ {valid_rate:.1f}%</div>',
            f'      <div style="font-size:11px; color:#475569;">{completed_actions} completed / {total_actions} attempted</div>',
            "    </div>",
            '    <div style="background:#fefce8; border:1px solid #fde047; border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">',
            '      <div style="font-size:11px; font-weight:700; color:#854d0e; text-transform:uppercase; letter-spacing:0.5px;">LLM Judge Overall</div>',
            f'      <div style="font-size:20px; font-weight:800; color:#713f12; margin:4px 0;">{judge_str}</div>',
            f'      <div style="font-size:11px; color:#475569;">Model: {judge_model}</div>',
            "    </div>",
            "  </div>",
            "",
            "  <!-- Detailed Objective Metric Panels -->",
            '  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px;">',
            "    <!-- Panel 1: Physical Execution & Coordination -->",
            '    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">',
            '      <div style="font-size:12.5px; font-weight:700; color:#0f172a; margin-bottom:8px; display:flex; align-items:center; gap:6px;">',
            "        <span>🎯</span> Physical Execution &amp; Coordination",
            "      </div>",
            '      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px;">',
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Resource Conflicts</div>',
            f'          <div style="font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;">{conflicts}</div>',
            '          <div style="font-size:9.5px; color:#64748b;">Lock contentions</div>',
            "        </div>",
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Dependency Errors</div>',
            f'          <div style="font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;">{dep_errors}</div>',
            '          <div style="font-size:9.5px; color:#64748b;">Prerequisite faults</div>',
            "        </div>",
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Navigation Moves</div>',
            f'          <div style="font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;">{total_moves}</div>',
            '          <div style="font-size:9.5px; color:#64748b;">Station transits</div>',
            "        </div>",
            "      </div>",
            '      <div style="margin-top:8px; font-size:10.5px; color:#475569;">',
            '        <b style="color:#0f172a;">Workload Shares:</b>',
            f'        <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:3px;">{workload_html}</div>',
            "      </div>",
            "    </div>",
            "",
            "    <!-- Panel 2: Natural-Language Communication -->",
            '    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">',
            '      <div style="font-size:12.5px; font-weight:700; color:#0f172a; margin-bottom:8px; display:flex; align-items:center; gap:6px;">',
            "        <span>💬</span> Natural-Language Communication",
            "      </div>",
            '      <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">',
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Message Volume</div>',
            f'          <div style="font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;">{num_msgs} msgs</div>',
            f'          <div style="font-size:9.5px; color:#64748b;">{total_tokens} tokens ({avg_tokens:.1f}/msg)</div>',
            "        </div>",
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Grounding Rate</div>',
            f'          <div style="font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;">{grounded_pct:.1f}%</div>',
            f'          <div style="font-size:9.5px; color:#64748b;">{ungrounded_cnt} ungrounded claims</div>',
            "        </div>",
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Non-Redundancy</div>',
            f'          <div style="font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;">{non_redundant_pct:.1f}%</div>',
            f'          <div style="font-size:9.5px; color:#64748b;">{redundant_cnt} repetitive msgs</div>',
            "        </div>",
            '        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:7px 9px;">',
            '          <div style="font-size:10px; color:#64748b; font-weight:600;">Run &amp; Condition</div>',
            f'          <div style="font-size:13px; font-weight:700; color:#0f172a; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{condition_name}</div>',
            f'          <div style="font-size:9.5px; color:#64748b; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{task_name}</div>',
            "        </div>",
            "      </div>",
            '      <div style="margin-top:8px; font-size:10.5px; color:#475569;">',
            '        <b style="color:#0f172a;">Speech Act Distribution:</b>',
            f'        <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:3px;">{intents_html}</div>',
            "      </div>",
            "    </div>",
            "  </div>",
            "",
            "  <!-- Collapsible Raw JSON Data -->",
            '  <details style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:7px 12px; margin-top:2px;">',
            '    <summary style="cursor:pointer; font-size:11px; font-weight:700; color:#475569;">',
            "      🔍 Click to expand raw JSON metrics payload",
            "    </summary>",
            f'    <pre style="margin-top:6px; font-size:10.5px; color:#0f172a; background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:8px; max-height:220px; overflow-y:auto; white-space:pre-wrap;">{raw_json_str}</pre>',
            "  </details>",
            "</div>",
        ]
        return "\n".join(html)


CUSTOM_CSS = """
/* Ensure high-contrast text and clear readability across light and dark Gradio modes */
.gradio-container {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
"""


# ---------------------------------------------------------
# 3. Gradio Replay Dashboard Application
# ---------------------------------------------------------
def create_gradio_interface(runs_dir: str = "runs"):
    """
    Creates and returns a Gradio Blocks interface implementing all
    replay dashboard panels.
    """
    # Lazy-load Gradio so non-UI CLI runs do not require it
    import gradio as gr

    trace_files = TraceReplayEngine.list_trace_files(runs_dir=runs_dir)
    file_choices = [os.path.basename(p) for p in trace_files]
    default_file = file_choices[0] if file_choices else None

    def _load_and_render(
        trace_filename: str,
        timestep: int,
        sender_filter: str,
        recipient_filter: str,
        intent_filter: str,
        include_system: bool,
        action_status_filter: str,
    ):
        """
        Gradio reactive event callback triggered whenever the user changes the selected
        trace file, scrubs the timestep slider, or adjusts any message/action filter.
        Reconstructs the state at `timestep` and returns updated HTML, judge-score DataFrame, table, and JSON outputs.
        """
        if not trace_filename:
            empty = (
                "<div style='padding:16px; color:#1e293b; font-family:system-ui, sans-serif; background:#f8fafc; border-radius:8px; border:1px dashed #94a3b8;'>"
                "<b style='color:#0f172a;'>📁 No execution trace found in <code>runs/</code>.</b><br>"
                "<span style='color:#334155;'>To generate a trace, run a simulation first in another terminal window:</span><br>"
                "<code style='background:#e2e8f0; color:#0f172a; font-weight:600; padding:2px 6px; border-radius:4px;'>python NLP_project_main.py --task 1 --condition reference</code><br>"
                "<span style='color:#475569;'>Once the simulation finishes, restart or refresh this dashboard to inspect the replay.</span>"
                "</div>"
            )
            return (
                empty,
                empty,
                empty,
                empty,
                empty,
                [],
                [],
                TraceReplayEngine.build_judge_scores_dataframe(None),
                empty,
            )

        full_path = os.path.join(runs_dir, trace_filename)
        if not os.path.isfile(full_path):
            empty = f"<div style='padding:12px; color:#b91c1c; font-weight:600; font-family:system-ui, sans-serif; background:#fee2e2; border-radius:8px; border:1px solid #fca5a5;'>Trace file not found: {trace_filename}</div>"
            return (
                empty,
                empty,
                empty,
                empty,
                empty,
                [],
                [],
                TraceReplayEngine.build_judge_scores_dataframe(None),
                empty,
            )

        trace = TraceReplayEngine.load_trace(full_path)
        max_t = TraceReplayEngine.get_max_timestep(trace)
        clamped_t = max(0, min(int(timestep), max_t))
        state_at_t = TraceReplayEngine.get_state_at_timestep(trace, clamped_t)

        failure_banner_html = TraceReplayEngine.render_failure_banner_html(
            trace, clamped_t, state_at_t
        )

        grid_html = TraceReplayEngine.render_grid_html(trace, state_at_t)
        dag_html = TraceReplayEngine.render_dag_html(trace, state_at_t)
        profiles_html = TraceReplayEngine.render_robot_profiles_html(trace, state_at_t)

        msgs = TraceReplayEngine.filter_messages(
            trace=trace,
            max_timestep=clamped_t,
            sender_filter=sender_filter,
            recipient_filter=recipient_filter,
            intent_filter=intent_filter,
            include_system=include_system,
        )
        msgs_html = TraceReplayEngine.render_messages_html(msgs)

        actions_list, conflicts_list = TraceReplayEngine.filter_actions_and_conflicts(
            trace=trace,
            max_timestep=clamped_t,
            robot_filter="ALL",
            status_filter=action_status_filter,
        )
        action_rows = [
            [
                a["timestep"],
                a["robot_id"],
                a["status"],
                a["action_type"],
                a["category"],
                a["details"],
            ]
            for a in actions_list
        ]
        conflict_rows = [
            [
                c["timestep"],
                c["requesting_robot"],
                c["held_by_robot"],
                c["resource_type"],
                c["resource_name"],
                c["attempted_action"],
            ]
            for c in conflicts_list
        ]

        judge_df = TraceReplayEngine.build_judge_scores_dataframe(
            trace.get("llm_judge_report")
        )
        scorecard_html = TraceReplayEngine.render_metrics_cards_html(trace)
        return (
            failure_banner_html,
            grid_html,
            dag_html,
            profiles_html,
            msgs_html,
            action_rows,
            conflict_rows,
            judge_df,
            scorecard_html,
        )

    with gr.Blocks(title="Multi-Robot Coordination Replay Dashboard") as demo:
        gr.Markdown(
            "# 🤖 Decentralized Multi-Robot Coordination — Replay Dashboard\n"
            "Scrub through simulation timesteps ($t = 0 \\dots T$) to inspect the kitchen workstations, "
            "robot capability profiles, dynamic Task Dependency Graph (DAG), inter-agent messages, "
            "executed/rejected actions, resource conflicts, and Independent LLM-Judge scores."
        )

        with gr.Row():
            trace_dropdown = gr.Dropdown(
                choices=file_choices,
                value=default_file,
                label="Stored Execution Trace (runs/*.json)",
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Refresh Runs", scale=1)
            timestep_slider = gr.Slider(
                minimum=0,
                maximum=45,
                step=1,
                value=0,
                label="Timesteps (t = 0 ... T seconds)",
                scale=3,
            )

        def _refresh_choices():
            files = TraceReplayEngine.list_trace_files(runs_dir=runs_dir)
            choices = [os.path.basename(p) for p in files]
            val = choices[0] if choices else None
            return gr.update(choices=choices, value=val)

        refresh_btn.click(fn=_refresh_choices, inputs=[], outputs=[trace_dropdown])

        failure_banner = gr.HTML(label="Controlled Failure Injection Live Status")

        with gr.Row():
            grid_panel = gr.HTML(label="Kitchen Workstations Playback")
            dag_panel = gr.HTML(label="Dynamic Task Dependency Graph (DAG)")

        gr.Markdown("### 🦾 Robot Capability Profiles & Live Agent State")
        profiles_panel = gr.HTML()

        with gr.Tabs():
            with gr.TabItem("💬 Inter-Agent Message Log"):
                with gr.Row():
                    sender_dd = gr.Dropdown(
                        choices=["ALL", "Robot_1", "Robot_2", "Robot_3", "System"],
                        value="ALL",
                        label="Sender",
                    )
                    recip_dd = gr.Dropdown(
                        choices=["ALL", "Robot_1", "Robot_2", "Robot_3"],
                        value="ALL",
                        label="Recipient",
                    )
                    intent_dd = gr.Dropdown(
                        choices=["ALL"] + [i.value for i in CommunicationIntent],
                        value="ALL",
                        label="Communication Intent",
                    )
                    sys_chk = gr.Checkbox(value=True, label="Include System Messages")
                msgs_panel = gr.HTML()

            with gr.TabItem("⚙️ Action History & Resource Conflicts"):
                status_dd = gr.Dropdown(
                    choices=[
                        "ALL",
                        "COMPLETED_ONLY",
                        "REJECTED_ONLY",
                        "INJECTED_FAILURES_ONLY",
                        "NAVIGATION_AND_CONFLICTS",
                    ],
                    value="ALL",
                    label="Action Filter",
                )
                actions_table = gr.Dataframe(
                    headers=[
                        "Timestep (s)",
                        "Robot",
                        "Status",
                        "Action",
                        "Category",
                        "Details",
                    ],
                    label="Executed & Rejected Actions",
                )
                conflicts_table = gr.Dataframe(
                    headers=[
                        "Timestep (s)",
                        "Requesting Robot",
                        "Locked By",
                        "Resource Type",
                        "Resource Name",
                        "Attempted Action",
                    ],
                    label="Mutual-Exclusion Resource Conflicts",
                )

            with gr.TabItem("📊 Final Metrics Scorecard & LLM-Judge Scores"):
                with gr.Row():
                    judge_plot = gr.BarPlot(
                        x="Dimension",
                        y="Score",
                        y_lim=[0, 5],
                        title="LLM Judge Scores (1-5)",
                        label="LLM Judge Scores",
                        scale=1,
                    )
                    scorecard_cards = gr.HTML(
                        label="Objective Performance & Communication Metrics Cards",
                        scale=2,
                    )

        inputs = [
            trace_dropdown,
            timestep_slider,
            sender_dd,
            recip_dd,
            intent_dd,
            sys_chk,
            status_dd,
        ]
        outputs = [
            failure_banner,
            grid_panel,
            dag_panel,
            profiles_panel,
            msgs_panel,
            actions_table,
            conflicts_table,
            judge_plot,
            scorecard_cards,
        ]

        for ctrl in inputs:
            ctrl.change(fn=_load_and_render, inputs=inputs, outputs=outputs)
        demo.load(fn=_load_and_render, inputs=inputs, outputs=outputs)
        if hasattr(gr, "Timer"):
            timer = gr.Timer(2)
            timer.tick(fn=_load_and_render, inputs=inputs, outputs=outputs)

    return demo


def launch_dashboard(
    runs_dir: str = "runs",
    port: int = 7860,
    inbrowser: bool = True,
    prevent_thread_lock: bool = False,
):
    """
    Launches the interactive Trace Replay Dashboard web server using Gradio.

    Parameters:
    - `runs_dir`: Directory containing execution trace JSON files (default: 'runs').
    - `port`: Local port to bind the Gradio web server to (default: 7860).
    - `inbrowser`: Whether to automatically open the default web browser upon startup.
    - `prevent_thread_lock`: Whether to prevent blocking the main thread (used when launching
      alongside live simulations).
    """
    demo = create_gradio_interface(runs_dir=runs_dir)
    print(f"\n=======================================================")
    print(f"🚀 Gradio Replay Dashboard is starting!")
    print(f"👉 Opening web browser at: http://localhost:{port}")
    if not prevent_thread_lock:
        print(f"Press Ctrl+C in this terminal to stop the web server.")
    print(f"=======================================================\n")
    return demo.launch(
        server_port=port,
        inbrowser=inbrowser,
        prevent_thread_lock=prevent_thread_lock,
        css=CUSTOM_CSS,
    )


__all__ = [
    "TraceReplayEngine",
    "create_gradio_interface",
    "launch_dashboard",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Robot Coordination Replay Dashboard (Gradio)"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing execution trace JSON files (default: runs)",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for Gradio server (default: 7860)"
    )
    args = parser.parse_args()

    launch_dashboard(runs_dir=args.runs_dir, port=args.port)
