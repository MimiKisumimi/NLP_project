"""
Passive observer, independent LLM Judge, and human-judge agreement verification (`observer_judge.py`).

Implements non-intrusive execution trace recording (`PassiveObserver`), 7-dimension semantic
communication evaluation (`LLMJudge`), and human-judge agreement verification (`HumanJudgeAgreementVerifier`).
"""

import os
import sys
import json
import math
import time
import uuid
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from models import CommunicationIntent, Message
from agent import LLMProvider, GeminiProvider, extract_json_from_text
from metrics import TaskPerformanceMetrics, CommunicationMetrics, MetricsEngine


def clamp_likert_score(
    value: Any, min_val: int = 1, max_val: int = 5, default: int = 3
) -> int:
    """Clamps a numeric or string value to an integer Likert scale [min_val, max_val]."""
    try:
        return max(min_val, min(max_val, int(round(float(value)))))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------
# Structured Schemas for Independent LLM Judge
# ---------------------------------------------------------
class JudgeDimensionScore(BaseModel):
    """1–5 rubric score and short textual justification for a single semantic communication dimension."""

    score: int = Field(
        ge=1,
        le=5,
        description="Integer score on a 1 to 5 scale (1 = Very Poor, 3 = Adequate, 5 = Excellent).",
    )
    justification: str = Field(
        description="Short textual justification citing specific evidence from the execution trace.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> int:
        """Pydantic pre-validator that clamps raw LLM judge score outputs to the integer range `[1, 5]`."""
        return clamp_likert_score(v)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `JudgeDimensionScore` (score and justification) into a dictionary."""
        return {
            "score": self.score,
            "justification": self.justification,
        }


class LLMJudgeResponseSchema(BaseModel):
    """Pydantic schema enforced on the independent LLM Judge response."""

    relevance: JudgeDimensionScore = Field(
        description="1. Relevance (1-5): Is communication on-topic and necessary for task completion?",
    )
    clarity: JudgeDimensionScore = Field(
        description="2. Clarity (1-5): Are messages unambiguous, concise, and easy for peers to interpret?",
    )
    grounding: JudgeDimensionScore = Field(
        description="3. Grounding (1-5): Are statements factually consistent with the physical environment state?",
    )
    coordination_utility: JudgeDimensionScore = Field(
        description="4. Coordination Utility (1-5): Did messages facilitate task allocation, handoffs, and avoid deadlocks?",
    )
    redundancy: JudgeDimensionScore = Field(
        description="5. Redundancy (1-5): Freedom from repeated/trivial chatter (5 = no redundant chatter, 1 = severe repetition).",
    )
    planning_coherence: JudgeDimensionScore = Field(
        description="6. Planning Coherence (1-5): Do proposed plans and commitments respect task dependencies and robot capabilities?",
    )
    failure_explanation_quality: JudgeDimensionScore = Field(
        description="7. Quality of Failure Explanations (1-5): Did agents clearly articulate errors, conflicts, or recovery plans?",
    )
    overall_summary: str = Field(
        description="Brief holistic summary of team communication quality across the episode.",
    )


@dataclass
class LLMJudgeReport:
    """Complete structured evaluation report produced by the Independent LLM Judge for a single run."""

    run_id: str
    task_name: str
    condition_name: str
    judge_model: str
    evaluation_mode: str  # "llm_judge"
    timestamp: str

    # 7 semantic communication dimensions (1-5 scale + justification)
    relevance: Dict[str, Any]
    clarity: Dict[str, Any]
    grounding: Dict[str, Any]
    coordination_utility: Dict[str, Any]
    redundancy: Dict[str, Any]
    planning_coherence: Dict[str, Any]
    failure_explanation_quality: Dict[str, Any]

    overall_average_score: float
    overall_summary: str

    def get_scores_dict(self) -> Dict[str, int]:
        """Returns a flat map of the 7 dimension names to their integer 1-5 scores."""
        return {
            "relevance": int(self.relevance.get("score", 3)),
            "clarity": int(self.clarity.get("score", 3)),
            "grounding": int(self.grounding.get("score", 3)),
            "coordination_utility": int(self.coordination_utility.get("score", 3)),
            "redundancy": int(self.redundancy.get("score", 3)),
            "planning_coherence": int(self.planning_coherence.get("score", 3)),
            "failure_explanation_quality": int(
                self.failure_explanation_quality.get("score", 3)
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the full `LLMJudgeReport` and all 7 dimension evaluations into a dictionary."""
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "condition_name": self.condition_name,
            "judge_model": self.judge_model,
            "evaluation_mode": self.evaluation_mode,
            "timestamp": self.timestamp,
            "dimensions": {
                "relevance": self.relevance,
                "clarity": self.clarity,
                "grounding": self.grounding,
                "coordination_utility": self.coordination_utility,
                "redundancy": self.redundancy,
                "planning_coherence": self.planning_coherence,
                "failure_explanation_quality": self.failure_explanation_quality,
            },
            "scores": self.get_scores_dict(),
            "overall_average_score": self.overall_average_score,
            "overall_summary": self.overall_summary,
        }


# ---------------------------------------------------------
# Passive Observer Module
# ---------------------------------------------------------
class PassiveObserver:
    """
    Non-intrusive passive observer that records complete execution traces without
    communicating with the agents or influencing execution.

    Receives and logs:
    - User instruction & task definition
    - Condition configuration & robot capability profiles
    - Initial environment state
    - Per-timestep global environment states
    - Per-timestep agent observations
    - Actions and action results (executed & failed/rejected actions)
    - Agent messages (with sender, recipient, timestep, local_state, previous_action, intent)
    - Resource conflicts and constraint violations
    - Final task state, ground-truth verification, objective metrics, and LLM Judge report
    """

    def __init__(self, run_id: Optional[str] = None, output_dir: str = "runs"):
        """Initializes trace storage buffers, incremental cursors, and wall-clock start time for `run_id`."""
        self.output_dir: str = output_dir
        self.run_id: str = run_id or self._generate_run_id()
        self.created_at: str = datetime.now().isoformat(timespec="seconds")

        # Core fields
        self.user_instruction: str = ""
        self.task_info: Dict[str, Any] = {}
        self.condition_config: Dict[str, Any] = {}
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        self.acting_agent_model: str = "unknown"
        self.initial_state: Dict[str, Any] = {}

        # Per-tick and cumulative event logs (deep-copied to remain strictly read-only)
        self.timestep_traces: List[Dict[str, Any]] = []
        self.global_states: List[Dict[str, Any]] = []
        self.agent_observations_history: List[Dict[str, Any]] = []
        self.actions_and_results: List[Dict[str, Any]] = []
        self.agent_messages: List[Dict[str, Any]] = []
        self.resource_conflicts: List[Dict[str, Any]] = []
        self.constraint_violations: List[Dict[str, Any]] = []

        # Final outcomes
        self.final_task_state: Dict[str, Any] = {}
        self.objective_metrics: Dict[str, Any] = {}
        self.failure_injections: Dict[str, Any] = {}
        self.llm_judge_report: Optional[Dict[str, Any]] = None
        self.execution_telemetry: Dict[str, Any] = {}
        self._start_perf_time: float = time.perf_counter()

        # Cursors for incremental tracking across timesteps
        self._last_completed_idx: int = 0
        self._last_failed_idx: int = 0
        self._last_msg_idx: int = 0
        self._last_conflict_idx: int = 0

    @staticmethod
    def _generate_run_id(prefix: str = "run") -> str:
        """Generates a timestamped, collision-resistant run identifier (e.g., `run_20261023_155500_a1b2c3`)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uid = uuid.uuid4().hex[:6]
        return f"{prefix}_{ts}_{short_uid}"

    def observe_initialization(self, env: Any, user_instruction: Optional[str] = None):
        """
        Records initial environment state, user instruction, condition config, and robot capability profiles
        at t=0 before execution starts. Does not mutate `env` or any agent.
        """
        task = getattr(env, "current_task", None)
        if user_instruction:
            self.user_instruction = user_instruction
        elif task:
            dishes_str = ", ".join(task.target_dishes)
            self.user_instruction = (
                f"{task.name} ({task.difficulty}): {task.description} "
                f"Deliver [{dishes_str}] to {task.target_station}."
            )
        else:
            self.user_instruction = "Complete the kitchen task."

        self.task_info = json.loads(json.dumps(task.to_dict())) if task else {}
        cond_cfg = getattr(env, "condition_config", None)
        self.condition_config = (
            json.loads(json.dumps(cond_cfg.to_dict())) if cond_cfg else {}
        )

        # Record robot profiles & acting model
        profiles: Dict[str, Dict[str, Any]] = {}
        detected_model = "unknown"
        for r_id, agent in getattr(env, "agents", {}).items():
            provider = getattr(agent, "llm_provider", None)
            model_name = getattr(
                provider,
                "model",
                provider.__class__.__name__ if provider else "unknown",
            )
            if detected_model == "unknown" and model_name:
                detected_model = str(model_name)
            profiles[r_id] = {
                "robot_id": r_id,
                "role": getattr(agent, "role", "General"),
                "description": getattr(agent, "role_description", ""),
                "skills": list(getattr(agent, "skills", [])),
                "carrying_capacity": getattr(agent, "carrying_capacity", 1),
                "tool_compatibility": list(getattr(agent, "tool_compatibility", [])),
                "memory_limit": getattr(agent, "memory_limit", None),
                "max_messages": getattr(agent, "max_messages", None),
                "max_comm_tokens": getattr(agent, "max_comm_tokens", None),
                "model": str(model_name),
            }
        self.agent_profiles = profiles
        self.acting_agent_model = detected_model

        # Deep-copy initial state
        self.initial_state = {
            "timestep": 0,
            "robot_locations": {
                r_id: str(loc)
                for r_id, loc in env.global_state.get("robot_locations", {}).items()
            },
            "items": {
                item_id: item.to_dict()
                for item_id, item in getattr(env, "items", {}).items()
            },
            "workstations": (
                list(getattr(env, "workstations", {}).keys())
            ),
            "task_progress": task.get_progress() if task else {},
        }

        # Reset incremental cursors
        self._last_completed_idx = len(getattr(env, "completed_actions", []))
        self._last_failed_idx = len(getattr(env, "failed_actions", []))
        self._last_msg_idx = len(getattr(env, "message_log", []))
        self._last_conflict_idx = len(getattr(env, "resource_conflicts", []))

    def observe_timestep(self, env: Any, timestep: int):
        """
        Passively records the global state, agent local observations, messages, executed/rejected
        actions, resource conflicts, and constraint violations after a simulation timestep.
        Never communicates with agents or modifies execution state.
        """
        # Global state snapshot at this timestep
        if getattr(env, "state_history", None):
            latest_global = json.loads(json.dumps(env.state_history[-1], default=str))
        else:
            latest_global = {
                "timestep": timestep,
                "robot_locations": {
                    r_id: str(loc)
                    for r_id, loc in env.global_state.get("robot_locations", {}).items()
                },
                "items": {
                    k: it.to_dict() for k, it in getattr(env, "items", {}).items()
                },
            }
        self.global_states.append(latest_global)

        # Agent local observations at this timestep
        tick_observations: Dict[str, Any] = {}
        for r_id, agent in getattr(env, "agents", {}).items():
            obs_copy = json.loads(
                json.dumps(getattr(agent, "local_observation", {}), default=str)
            )
            tick_observations[r_id] = obs_copy
        self.agent_observations_history.append(
            {
                "timestep": timestep,
                "observations": tick_observations,
            }
        )

        # 3. Newly completed actions & newly rejected/failed actions (constraint violations)
        all_completed = getattr(env, "completed_actions", [])
        new_completed = all_completed[self._last_completed_idx :]
        self._last_completed_idx = len(all_completed)

        all_failed = getattr(env, "failed_actions", [])
        new_failed = all_failed[self._last_failed_idx :]
        self._last_failed_idx = len(all_failed)

        tick_actions: List[Dict[str, Any]] = []
        for ca in new_completed:
            entry = {
                "timestep": ca.get("timestep", timestep),
                "robot_id": ca.get("robot_id"),
                "status": "completed",
                "action": json.loads(json.dumps(ca.get("action", {}), default=str)),
            }
            tick_actions.append(entry)
            self.actions_and_results.append(entry)

        for fa in new_failed:
            violation_entry = {
                "timestep": fa.get("timestep", timestep),
                "robot_id": fa.get("robot_id"),
                "status": "rejected",
                "function": fa.get("function"),
                "error": fa.get("error"),
            }
            tick_actions.append(violation_entry)
            self.actions_and_results.append(violation_entry)
            self.constraint_violations.append(violation_entry)

        # New messages routed during this timestep
        all_msgs: List[Message] = getattr(env, "message_log", [])
        new_msgs = all_msgs[self._last_msg_idx :]
        self._last_msg_idx = len(all_msgs)

        tick_messages: List[Dict[str, Any]] = []
        for m in new_msgs:
            m_dict = json.loads(json.dumps(m.to_dict(), default=str))
            m_dict["is_peer_message"] = m.sender != "System"
            tick_messages.append(m_dict)
            self.agent_messages.append(m_dict)

        # New resource conflicts during this timestep
        all_conflicts = getattr(env, "resource_conflicts", [])
        new_conflicts = all_conflicts[self._last_conflict_idx :]
        self._last_conflict_idx = len(all_conflicts)

        tick_conflicts: List[Dict[str, Any]] = []
        for rc in new_conflicts:
            rc_copy = json.loads(json.dumps(rc, default=str))
            tick_conflicts.append(rc_copy)
            self.resource_conflicts.append(rc_copy)

        # Append consolidated per-tick trace record
        self.timestep_traces.append(
            {
                "timestep": timestep,
                "global_state": latest_global,
                "agent_observations": tick_observations,
                "actions_and_results": tick_actions,
                "messages": tick_messages,
                "resource_conflicts": tick_conflicts,
            }
        )

    def finalize_observation(
        self,
        env: Any,
        task_metrics: Optional[TaskPerformanceMetrics] = None,
        comm_metrics: Optional[CommunicationMetrics] = None,
        judge_report: Optional[LLMJudgeReport] = None,
    ) -> Dict[str, Any]:
        """
        Records the final task state, ground-truth simulator verification, objective metrics,
        and optional LLM Judge evaluation report at the end of the episode.
        """
        if not self.initial_state:
            self.observe_initialization(env)

        # Catch up if env ran timesteps without per-tick hooks
        if not self.timestep_traces and getattr(env, "state_history", None):
            self._reconstruct_from_env_history(env)

        verif = env.verify_current_task()
        t_metrics = task_metrics or MetricsEngine.compute_task_metrics(env)
        c_metrics = comm_metrics or MetricsEngine.compute_communication_metrics(env)

        self.final_task_state = {
            "final_timestep": getattr(env, "current_time", 0),
            "verification": verif.to_dict(),
            "final_robot_locations": {
                r_id: str(loc)
                for r_id, loc in env.global_state.get("robot_locations", {}).items()
            },
            "final_items": {
                item_id: item.to_dict()
                for item_id, item in getattr(env, "items", {}).items()
            },
            "task_progress": (
                env.current_task.get_progress()
                if getattr(env, "current_task", None)
                else {}
            ),
        }

        self.objective_metrics = {
            "task_performance": t_metrics.to_dict(),
            "communication": c_metrics.to_dict(),
        }

        # Record failure injections summary from FailureManager if available
        if getattr(env, "failure_manager", None):
            self.failure_injections = env.failure_manager.get_summary()

        # Collect processing time, API latency, token usage, and estimated financial cost
        processing_time_s = round(
            max(0.0, time.perf_counter() - self._start_perf_time), 4
        )
        total_calls = 0
        total_latency_ms = 0.0
        prompt_toks = 0
        comp_toks = 0
        total_toks = 0
        est_cost = 0.0
        decoding_params: Dict[str, Any] = {}
        seen_providers: set = set()

        for agent in getattr(env, "agents", {}).values():
            prov = getattr(agent, "llm_provider", None)
            if (
                prov
                and id(prov) not in seen_providers
                and hasattr(prov, "get_usage_stats")
            ):
                seen_providers.add(id(prov))
                stats = prov.get_usage_stats()
                total_calls += int(stats.get("call_count", 0))
                total_latency_ms += float(stats.get("total_latency_ms", 0.0))
                prompt_toks += int(stats.get("prompt_tokens", 0))
                comp_toks += int(stats.get("completion_tokens", 0))
                total_toks += int(stats.get("total_tokens", 0))
                est_cost += float(stats.get("estimated_cost_usd", 0.0))
                if not decoding_params and stats.get("decoding_params"):
                    decoding_params = dict(stats["decoding_params"])

        self.execution_telemetry = {
            "processing_time_sec": processing_time_s,
            "api_call_count": total_calls,
            "total_api_latency_ms": round(total_latency_ms, 2),
            "mean_api_latency_ms": (
                round(total_latency_ms / total_calls, 2) if total_calls > 0 else 0.0
            ),
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_api_tokens": total_toks,
            "estimated_cost_usd": round(est_cost, 6),
            "decoding_parameters": decoding_params,
            "robot_count": len(getattr(env, "agents", {})),
        }

        if judge_report is not None:
            self.llm_judge_report = judge_report.to_dict()

        return self.to_trace_dict()

    def _reconstruct_from_env_history(self, env: Any):
        """Reconstructs trace records from `env.state_history` if observer was attached post-run."""
        self.global_states = [
            json.loads(json.dumps(snap, default=str))
            for snap in getattr(env, "state_history", [])
        ]
        self.agent_messages = []
        for m in getattr(env, "message_log", []):
            m_dict = json.loads(json.dumps(m.to_dict(), default=str))
            m_dict["is_peer_message"] = m.sender != "System"
            self.agent_messages.append(m_dict)

        self.actions_and_results = []
        for ca in getattr(env, "completed_actions", []):
            self.actions_and_results.append(
                {
                    "timestep": ca.get("timestep", 0),
                    "robot_id": ca.get("robot_id"),
                    "status": "completed",
                    "action": json.loads(json.dumps(ca.get("action", {}), default=str)),
                }
            )

        self.constraint_violations = []
        for fa in getattr(env, "failed_actions", []):
            viol = {
                "timestep": fa.get("timestep", 0),
                "robot_id": fa.get("robot_id"),
                "status": "rejected",
                "function": fa.get("function"),
                "error": fa.get("error"),
            }
            self.actions_and_results.append(viol)
            self.constraint_violations.append(viol)

        self.resource_conflicts = [
            json.loads(json.dumps(rc, default=str))
            for rc in getattr(env, "resource_conflicts", [])
        ]

    def attach_judge_report(self, judge_report: LLMJudgeReport):
        """Attaches an LLMJudgeReport to the trace."""
        self.llm_judge_report = judge_report.to_dict()

    def to_trace_dict(self) -> Dict[str, Any]:
        """Returns the complete structured execution trace dictionary."""
        peer_messages = [
            m
            for m in self.agent_messages
            if m.get("is_peer_message", m.get("sender") != "System")
        ]
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "user_instruction": self.user_instruction,
            "task": self.task_info,
            "condition_config": self.condition_config,
            "acting_agent_model": self.acting_agent_model,
            "agent_profiles": self.agent_profiles,
            "initial_state": self.initial_state,
            "timestep_traces": self.timestep_traces,
            "global_environment_states": self.global_states,
            "agent_observations": self.agent_observations_history,
            "actions_and_results": self.actions_and_results,
            "agent_messages": self.agent_messages,
            "peer_messages": peer_messages,
            "resource_conflicts": self.resource_conflicts,
            "constraint_violations": self.constraint_violations,
            "final_task_state": self.final_task_state,
            "objective_metrics": self.objective_metrics,
            "failure_injections": self.failure_injections,
            "execution_telemetry": self.execution_telemetry,
            "llm_judge_report": self.llm_judge_report,
        }

    def export_trace(self, filepath: Optional[str] = None) -> str:
        """
        Exports the complete execution trace to structured JSON (`runs/<run_id>.json`).
        Returns the saved file path.
        """
        if filepath is None:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, f"{self.run_id}.json")
        else:
            parent = os.path.dirname(filepath)
            if parent:
                os.makedirs(parent, exist_ok=True)

        trace_data = self.to_trace_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)

        return filepath


# ---------------------------------------------------------
# Independent LLM Judge Pipeline
# ---------------------------------------------------------
class LLMJudge:
    """
    Independent LLM Judge for semantic communication evaluation.

    Evaluates 7 semantic qualities on a 1–5 scale with short justifications:
    1. Relevance (Is communication on-topic and necessary?)
    2. Clarity (Are messages unambiguous?)
    3. Grounding (Are statements factually consistent with the physical environment?)
    4. Coordination Utility (Did messages facilitate action and avoid deadlocks?)
    5. Redundancy (Is there repeated or trivial chatter? 5 = no redundancy, 1 = high redundancy)
    6. Planning Coherence (Do proposed plans respect task dependencies?)
    7. Quality of Failure Explanations (Did the agent clearly articulate errors?)

    Model Independence:
    - Uses gemini-3.1-pro-preview as an independent LLM Judge distinct from acting agents (e.g., gemini-3.5-flash-lite).
    """

    DIMENSIONS: List[str] = [
        "relevance",
        "clarity",
        "grounding",
        "coordination_utility",
        "redundancy",
        "planning_coherence",
        "failure_explanation_quality",
    ]

    def __init__(
        self,
        judge_model: Optional[str] = "gemini-3.1-pro-preview",
        acting_agent_model: Optional[str] = None,
        timeout: float = 300.0,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the Independent LLM Judge for semantic communication evaluation.

        Parameters:
        - `judge_model`: Independent Gemini model name for judging (default: 'gemini-3.1-pro-preview').
        - `acting_agent_model`: Model name used by acting agents (e.g. 'gemini-3.5-flash-lite')
          tracked to verify model independence.
        - `timeout`: Timeout in seconds for HTTP requests (default: 300.0s).
        - `api_key`: Optional Gemini API key (defaults to GEMINI_API_KEY environment variable).
        """
        self.judge_model = judge_model or os.getenv(
            "JUDGE_LLM_MODEL", "gemini-3.1-pro-preview"
        )
        self.acting_agent_model = acting_agent_model
        self.timeout = timeout
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def _build_judge_prompts(self, trace: Dict[str, Any]) -> Tuple[str, str]:
        """Constructs the system instruction and trace summary prompt for the LLM Judge."""
        system_instruction = (
            "You are an Independent LLM Judge evaluating multi-robot natural-language coordination "
            "in a decentralized kitchen simulation.\n"
            "You did NOT participate in the simulation. Your role is strictly to evaluate the semantic quality "
            "of inter-agent communication based on the ground-truth execution trace provided.\n\n"
            "Score each of the 7 dimensions on an integer scale from 1 to 5 and provide a concise, evidence-based justification:\n"
            "1. relevance (1-5): Is communication on-topic, task-focused, and necessary? (1 = off-topic/noisy, 5 = highly relevant and purposeful)\n"
            "2. clarity (1-5): Are messages unambiguous, well-structured, and easy for peer robots to act upon? (1 = vague/confusing, 5 = crystal clear)\n"
            "3. grounding (1-5): Are statements factually consistent with the physical simulator state (item locations, states, locks)? "
            "(1 = frequent hallucinations/false claims, 5 = 100% grounded in physical truth)\n"
            "4. coordination_utility (1-5): Did messages facilitate task allocation, capability discovery, handoffs, and deadlock avoidance? "
            "(1 = no coordination value, 5 = directly enabled smooth multi-robot synergy)\n"
            "5. redundancy (1-5): Freedom from repetitive or trivial chatter. "
            "(5 = concise with zero redundant repetition; 3 = occasional repetition; 1 = severe spam/duplicate messages)\n"
            "6. planning_coherence (1-5): Do proposed plans, commitments, and requests respect the task dependency graph (e.g. cut before assemble, cook before serve) "
            "and heterogeneous robot capabilities? (1 = incoherent/violates dependencies, 5 = logically sound plan)\n"
            "7. failure_explanation_quality (1-5): When actions failed, resources were locked, or failures were injected, did agents clearly articulate the cause "
            "and propose recovery? If no failures occurred, evaluate how clearly agents communicated status/warnings to prevent errors. "
            "(1 = silent or misleading on errors, 5 = clear diagnosis and constructive replanning)\n\n"
            "Respond strictly with valid JSON matching the requested schema. Expected JSON structure:\n"
            "{\n"
            '  "relevance": {"score": 4, "justification": "Communication stayed focused on task goals."},\n'
            '  "clarity": {"score": 5, "justification": "Clear instructions and messages with explicit details."},\n'
            '  "grounding": {"score": 4, "justification": "Statements were grounded in physical reality and actual item states."},\n'
            '  "coordination_utility": {"score": 4, "justification": "Facilitated effective task allocation and prevented deadlocks."},\n'
            '  "redundancy": {"score": 5, "justification": "Concise communication without unnecessary chatter."},\n'
            '  "planning_coherence": {"score": 4, "justification": "Proposals and actions respected task dependencies."},\n'
            '  "failure_explanation_quality": {"score": 4, "justification": "Clear diagnostic feedback and prompt error warnings."},\n'
            '  "overall_summary": "Effective coordination and clear communication throughout the task."\n'
            "}"
        )

        task_info = trace.get("task", {})
        cond_info = trace.get("condition_config", {})
        final_state = trace.get("final_task_state", {})
        obj_metrics = trace.get("objective_metrics", {})
        comm_m = obj_metrics.get("communication", {})
        task_m = obj_metrics.get("task_performance", {})

        peer_msgs = trace.get("peer_messages", [])
        if not peer_msgs:
            peer_msgs = [
                m
                for m in trace.get("agent_messages", [])
                if m.get("sender") != "System"
            ]

        msg_lines = []
        for idx, m in enumerate(peer_msgs[:40], start=1):
            t = m.get("timestep", 0)
            sender = m.get("sender", "?")
            rec = m.get("recipient", "ALL")
            intent = m.get("intent") or m.get("function") or "STATUS_UPDATE"
            text = m.get("message_content") or m.get("content") or ""
            prev_act = m.get("previous_action", "NONE")
            loc = m.get("local_state", {}).get("location", "?")
            msg_lines.append(
                f'  [{idx}] t={t}s | {sender} (at {loc}, prev={prev_act}) -> {rec} [{intent}]: "{text}"'
            )
        msgs_block = (
            "\n".join(msg_lines)
            if msg_lines
            else "  (No peer-to-peer messages sent during this run.)"
        )

        violations = trace.get("constraint_violations", [])
        viol_lines = [
            f"  - t={v.get('timestep')}s {v.get('robot_id')} failed {v.get('function')}: {v.get('error')}"
            for v in violations[:15]
        ]
        viol_block = "\n".join(viol_lines) if viol_lines else "  (None)"

        conflicts = trace.get("resource_conflicts", [])
        conf_lines = [
            f"  - t={c.get('timestep')}s {c.get('robot_id')} conflicted with {c.get('conflicted_with')} on {c.get('resource_type')} '{c.get('resource_name')}' ({c.get('attempted_action')})"
            for c in conflicts[:15]
        ]
        conf_block = "\n".join(conf_lines) if conf_lines else "  (None)"

        ungrounded_list = comm_m.get("ungrounded_examples", [])
        ungrounded_block = (
            "\n".join([f"  - {ex}" for ex in ungrounded_list])
            if ungrounded_list
            else "  (None detected by simulator)"
        )

        user_prompt = (
            f"=== EXECUTION TRACE FOR INDEPENDENT JUDGE EVALUATION ===\n"
            f"Run ID: {trace.get('run_id', 'unknown')}\n"
            f"User Instruction: {trace.get('user_instruction', '')}\n"
            f"Task: {task_info.get('name', 'Unknown')} (Difficulty: {task_info.get('difficulty', 'Unknown')}, Target: {task_info.get('target_dishes', [])})\n"
            f"Limitation Condition: {cond_info.get('name', 'Unknown')} - {cond_info.get('description', '')}\n"
            f"Robot Profiles: {json.dumps(trace.get('agent_profiles', {}), default=str)}\n\n"
            f"--- SIMULATOR GROUND-TRUTH OUTCOME & FACTUAL METRICS ---\n"
            f"Task Completed (Ground Truth): {final_state.get('verification', {}).get('is_complete', False)}\n"
            f"DAG Progress: {final_state.get('task_progress', {}).get('completed_nodes', 0)}/{final_state.get('task_progress', {}).get('total_nodes', 0)} steps\n"
            f"Makespan: {task_m.get('makespan', 0)}s | Valid Action Rate: {task_m.get('valid_action_rate', 0.0)} | Recovery Rate: {task_m.get('recovery_rate', 1.0)}\n"
            f"Total Peer Messages: {comm_m.get('number_of_messages', len(peer_msgs))} | Grounded Rate: {comm_m.get('grounded_message_rate', 1.0)} | Redundant Rate: {comm_m.get('redundant_message_rate', 0.0)}\n"
            f"Ungrounded Claims Flagged by Simulator:\n{ungrounded_block}\n\n"
            f"--- CONSTRAINT VIOLATIONS / REJECTED ACTIONS ---\n{viol_block}\n\n"
            f"--- RESOURCE CONFLICTS ---\n{conf_block}\n\n"
            f"--- INTER-AGENT PEER MESSAGE LOG ---\n{msgs_block}\n\n"
            f"Evaluate all 7 semantic communication dimensions on a 1-5 scale with short justifications."
        )
        return system_instruction, user_prompt

    def _call_gemini_judge(
        self,
        system_instruction: str,
        user_prompt: str,
        max_retries: int = 5,
        initial_backoff: float = 5.0,
    ) -> Optional[LLMJudgeResponseSchema]:
        """Queries the Gemini API using the GeminiProvider structured output pipeline with retry logic."""
        model_name = self.judge_model.replace(" (independent-judge-prompt)", "").strip()
        provider = GeminiProvider(
            api_key=self.api_key,
            model=model_name,
            timeout=self.timeout,
            temperature=0.1,
        )
        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
            try:
                result = provider.generate_structured_output(
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    schema=LLMJudgeResponseSchema,
                )
                if result is not None:
                    return result
                out_stream = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
                print(
                    f"[LLMJudge] Attempt {attempt}/{max_retries} returned no result (rate limit, timeout, or parse failure). "
                    f"Retrying in {backoff:.1f}s...",
                    file=out_stream,
                    flush=True,
                )
            except Exception as exc:
                out_stream = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
                print(
                    f"[LLMJudge] Attempt {attempt}/{max_retries} failed with error: {exc}. "
                    f"Retrying in {backoff:.1f}s...",
                    file=out_stream,
                    flush=True,
                )

            if attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)

        return None

    def evaluate_trace(
        self,
        trace: Union[Dict[str, Any], PassiveObserver],
    ) -> LLMJudgeReport:
        """
        Evaluates a completed execution trace (or PassiveObserver instance) on all 7 semantic
        communication qualities (1-5 scale + short justification) and returns an `LLMJudgeReport`.
        If the observer is passed directly, also attaches the report to the observer.
        """
        observer_ref: Optional[PassiveObserver] = None
        if isinstance(trace, PassiveObserver):
            observer_ref = trace
            trace_dict = trace.to_trace_dict()
        else:
            trace_dict = trace

        sys_prompt, user_prompt = self._build_judge_prompts(trace_dict)

        parsed_response = self._call_gemini_judge(sys_prompt, user_prompt)
        if parsed_response is None:
            raise RuntimeError(
                f"[LLMJudge] Failed to obtain evaluation from Gemini model '{self.judge_model}' "
                "after all retry attempts. Check API key, network connection, or quota limits."
            )
        eval_mode = "llm_judge"

        scores = [
            parsed_response.relevance.score,
            parsed_response.clarity.score,
            parsed_response.grounding.score,
            parsed_response.coordination_utility.score,
            parsed_response.redundancy.score,
            parsed_response.planning_coherence.score,
            parsed_response.failure_explanation_quality.score,
        ]
        overall_avg = round(sum(scores) / len(scores), 2)

        report = LLMJudgeReport(
            run_id=trace_dict.get("run_id", "unknown_run"),
            task_name=trace_dict.get("task", {}).get("name", "Unknown Task"),
            condition_name=trace_dict.get("condition_config", {}).get(
                "name", "Unknown Condition"
            ),
            judge_model=self.judge_model,
            evaluation_mode=eval_mode,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            relevance=parsed_response.relevance.to_dict(),
            clarity=parsed_response.clarity.to_dict(),
            grounding=parsed_response.grounding.to_dict(),
            coordination_utility=parsed_response.coordination_utility.to_dict(),
            redundancy=parsed_response.redundancy.to_dict(),
            planning_coherence=parsed_response.planning_coherence.to_dict(),
            failure_explanation_quality=parsed_response.failure_explanation_quality.to_dict(),
            overall_average_score=overall_avg,
            overall_summary=parsed_response.overall_summary
            or f"Average Judge Score: {overall_avg}/5.0",
        )

        if observer_ref is not None:
            observer_ref.attach_judge_report(report)

        return report

    @staticmethod
    def format_judge_report(report: LLMJudgeReport) -> str:
        """Formats the 7-dimension LLM Judge evaluation report for console display."""
        lines = [
            "=" * 70,
            "INDEPENDENT LLM JUDGE SEMANTIC COMMUNICATION REPORT",
            "=" * 70,
            f"Run ID:              {report.run_id}",
            f"Task / Condition:    {report.task_name} | {report.condition_name}",
            f"Judge Model:         {report.judge_model} | Mode: {report.evaluation_mode}",
            "-" * 70,
            f"1. Relevance                   [{report.relevance['score']}/5]: {report.relevance['justification']}",
            f"2. Clarity                     [{report.clarity['score']}/5]: {report.clarity['justification']}",
            f"3. Grounding                   [{report.grounding['score']}/5]: {report.grounding['justification']}",
            f"4. Coordination Utility        [{report.coordination_utility['score']}/5]: {report.coordination_utility['justification']}",
            f"5. Redundancy (No Chatter)     [{report.redundancy['score']}/5]: {report.redundancy['justification']}",
            f"6. Planning Coherence          [{report.planning_coherence['score']}/5]: {report.planning_coherence['justification']}",
            f"7. Failure Explanation Quality [{report.failure_explanation_quality['score']}/5]: {report.failure_explanation_quality['justification']}",
            "-" * 70,
            f"Overall Judge Score: {report.overall_average_score:.2f} / 5.00",
            f"Summary:             {report.overall_summary}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------
# Human-Judge Agreement Verification
# ---------------------------------------------------------
class HumanJudgeAgreementVerifier:
    """
    Evaluates a sample of traces to check the reliability of the LLM judge,
    computing percentage agreement (exact and within-1-point) and Pearson correlation (r)
    between human and judge ratings across all 7 dimensions.
    """

    DIMENSIONS: List[str] = LLMJudge.DIMENSIONS

    @staticmethod
    def _pearson_r(x: List[float], y: List[float]) -> float:
        """Computes Pearson correlation coefficient r between two rating vectors."""
        n = len(x)
        if n < 2 or len(y) != n:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if den_x == 0.0 or den_y == 0.0:
            # If both have zero variance and identical values, correlation is 1.0
            return 1.0 if all(xi == yi for xi, yi in zip(x, y)) else 0.0
        return round(max(-1.0, min(1.0, num / (den_x * den_y))), 4)

    # ###Legacy metrics that were not used in the final analysis###
    @staticmethod
    def _cohens_kappa(
        human_ratings: List[int],
        judge_ratings: List[int],
        categories: Tuple[int, ...] = (1, 2, 3, 4, 5),
        weighted: bool = False,
    ) -> float:
        """
        Computes unweighted or quadratic-weighted Cohen's kappa (κ) for ordinal 1–5 ratings.
        """
        n = len(human_ratings)
        if n == 0 or len(judge_ratings) != n:
            return 0.0

        k = len(categories)
        cat_idx = {c: i for i, c in enumerate(categories)}

        # Confusion matrix O[i][j]
        observed = [[0.0 for _ in range(k)] for _ in range(k)]
        for h, j in zip(human_ratings, judge_ratings):
            hi = cat_idx.get(clamp_likert_score(h), 2)
            ji = cat_idx.get(clamp_likert_score(j), 2)
            observed[hi][ji] += 1.0

        # Row and column marginals
        row_totals = [sum(observed[i]) for i in range(k)]
        col_totals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

        # Expected matrix E[i][j]
        expected = [
            [(row_totals[i] * col_totals[j]) / float(n) for j in range(k)]
            for i in range(k)
        ]

        if not weighted:
            p_o = sum(observed[i][i] for i in range(k)) / float(n)
            p_e = sum(expected[i][i] for i in range(k)) / float(n)
            if abs(1.0 - p_e) < 1e-9:
                return 1.0 if abs(1.0 - p_o) < 1e-9 else 0.0
            return round((p_o - p_e) / (1.0 - p_e), 4)

        # Quadratic weights w_ij = ((i - j)^2) / ((k - 1)^2)
        denom_w = float((k - 1) ** 2)
        num_obs = 0.0
        num_exp = 0.0
        # Accumulate quadratic-weighted observed and expected disagreement across the k x k matrix
        for i in range(k):
            for j in range(k):
                w_ij = ((i - j) ** 2) / denom_w
                num_obs += w_ij * (observed[i][j] / float(n))
                num_exp += w_ij * (expected[i][j] / float(n))

        if abs(num_exp) < 1e-9:
            return 1.0 if abs(num_obs) < 1e-9 else 0.0
        return round(1.0 - (num_obs / num_exp), 4)

    @classmethod
    def sample_traces_for_annotation(
        cls,
        runs_dir: str = "runs",
        sample_size: int = 12,
        output_worksheet_path: str = "runs/human_judge_annotation_sample.json",
    ) -> List[Dict[str, Any]]:
        """
        Samples up to `sample_size` message traces from `runs_dir` and exports a structured JSON worksheet
        for manual human rating alongside the LLM Judge's scores.
        """
        sample_size = max(1, min(20, sample_size))
        loaded_traces: List[Dict[str, Any]] = []

        if os.path.isdir(runs_dir):
            for fname in sorted(os.listdir(runs_dir)):
                if (
                    fname.endswith(".json")
                    and "human_judge" not in fname
                    and "statistical" not in fname
                ):
                    fpath = os.path.join(runs_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict) and "run_id" in data:
                                loaded_traces.append(data)
                    except Exception:
                        pass

        judge = LLMJudge()
        items: List[Dict[str, Any]] = []

        for tr in loaded_traces[:sample_size]:
            j_rep = tr.get("llm_judge_report")
            if not j_rep:
                j_rep = judge.evaluate_trace(tr).to_dict()
            j_scores = j_rep.get("scores", {})
            peer_msgs = tr.get("peer_messages", [])
            msg_preview = [
                f"t={m.get('timestep')}s {m.get('sender')}->{m.get('recipient')} [{m.get('intent')}]: {m.get('content')}"
                for m in peer_msgs[:8]
            ]
            items.append(
                {
                    "sample_id": f"sample_{len(items) + 1:02d}",
                    "run_id": tr.get("run_id"),
                    "task_name": tr.get("task", {}).get("name", "Task"),
                    "condition_name": tr.get("condition_config", {}).get(
                        "name", "Condition"
                    ),
                    "message_trace_excerpt": msg_preview,
                    "llm_judge_scores": j_scores,
                    "llm_judge_justifications": j_rep.get("dimensions", {}),
                    "human_scores": dict(
                        j_scores
                    ),  # Pre-populated template for human annotator adjustment
                    "human_notes": "Review message_trace_excerpt and adjust 1-5 scores in human_scores if you disagree with llm_judge_scores.",
                }
            )

        if output_worksheet_path:
            parent = os.path.dirname(output_worksheet_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_worksheet_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"sample_size": len(items), "samples": items},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        return items

    @classmethod
    def compute_agreement(
        cls, annotated_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes per-dimension and pooled agreement metrics (exact agreement, within-1 agreement,
        and Pearson r) between `human_scores` and `llm_judge_scores`.
        """
        per_dimension: Dict[str, Dict[str, Any]] = {}
        all_human: List[int] = []
        all_judge: List[int] = []

        # Collect paired human and judge 1–5 ratings per dimension and across all dimensions
        for dim in cls.DIMENSIONS:
            h_list: List[int] = []
            j_list: List[int] = []
            for item in annotated_samples:
                h_map = item.get("human_scores", {})
                j_map = item.get("llm_judge_scores", {})
                if dim in h_map and dim in j_map:
                    h_val = clamp_likert_score(h_map[dim])
                    j_val = clamp_likert_score(j_map[dim])
                    h_list.append(h_val)
                    j_list.append(j_val)
                    all_human.append(h_val)
                    all_judge.append(j_val)

            n = len(h_list)
            if n == 0:
                continue

            exact_matches = sum(1 for h, j in zip(h_list, j_list) if h == j)
            within_one = sum(1 for h, j in zip(h_list, j_list) if abs(h - j) <= 1)
            # ###Legacy metrics that were not used in the final analysis###
            mae = sum(abs(j - h) for h, j in zip(h_list, j_list)) / float(n)
            bias = sum((j - h) for h, j in zip(h_list, j_list)) / float(n)

            per_dimension[dim] = {
                "n": n,
                "human_mean": round(sum(h_list) / float(n), 2),
                "judge_mean": round(sum(j_list) / float(n), 2),
                "exact_agreement_pct": round((exact_matches / float(n)) * 100.0, 1),
                "within_one_agreement_pct": round((within_one / float(n)) * 100.0, 1),
                "pearson_r": cls._pearson_r(
                    [float(x) for x in h_list], [float(y) for y in j_list]
                ),
                # ###Legacy metrics that were not used in the final analysis###
                "cohens_kappa_unweighted": cls._cohens_kappa(
                    h_list, j_list, weighted=False
                ),
                "cohens_kappa_weighted": cls._cohens_kappa(
                    h_list, j_list, weighted=True
                ),
                "mae": round(mae, 3),
                "mean_signed_bias": round(bias, 3),
            }

        total_pairs = len(all_human)
        if total_pairs > 0:
            pooled_exact = round(
                (
                    sum(1 for h, j in zip(all_human, all_judge) if h == j)
                    / float(total_pairs)
                )
                * 100.0,
                1,
            )
            pooled_within_1 = round(
                (
                    sum(1 for h, j in zip(all_human, all_judge) if abs(h - j) <= 1)
                    / float(total_pairs)
                )
                * 100.0,
                1,
            )
            pooled_r = cls._pearson_r(
                [float(x) for x in all_human], [float(y) for y in all_judge]
            )
            # ###Legacy metrics that were not used in the final analysis###
            pooled_kappa = cls._cohens_kappa(all_human, all_judge, weighted=False)
            pooled_kappa_w = cls._cohens_kappa(all_human, all_judge, weighted=True)
            pooled_mae = round(
                sum(abs(j - h) for h, j in zip(all_human, all_judge))
                / float(total_pairs),
                3,
            )
            pooled_bias = round(
                sum((j - h) for h, j in zip(all_human, all_judge)) / float(total_pairs),
                3,
            )
        else:
            pooled_exact = 0.0
            pooled_within_1 = 0.0
            pooled_r = 0.0
            pooled_kappa = 0.0
            pooled_kappa_w = 0.0
            pooled_mae = 0.0
            pooled_bias = 0.0

        return {
            "num_traces_evaluated": len(annotated_samples),
            "total_dimension_ratings": total_pairs,
            "pooled_metrics": {
                "exact_agreement_pct": pooled_exact,
                "within_one_agreement_pct": pooled_within_1,
                "pearson_r": pooled_r,
                "cohens_kappa_unweighted": pooled_kappa,
                "cohens_kappa_weighted": pooled_kappa_w,
                "mae": pooled_mae,
                "mean_signed_bias": pooled_bias,
            },
            "per_dimension_metrics": per_dimension,
        }

    @classmethod
    def format_agreement_report(cls, agreement_data: Dict[str, Any]) -> str:
        """Formats a clean summary table of Human vs. LLM Judge agreement."""
        pm = agreement_data.get("pooled_metrics", {})
        per_dim = agreement_data.get("per_dimension_metrics", {})

        lines = [
            "=" * 78,
            "HUMAN vs. INDEPENDENT LLM JUDGE RELIABILITY & AGREEMENT REPORT",
            "=" * 78,
            f"Traces Sampled: {agreement_data.get('num_traces_evaluated', 0)} | Total Ratings Compared: {agreement_data.get('total_dimension_ratings', 0)}",
            f"Pooled Exact Agreement:     {pm.get('exact_agreement_pct', 0.0):.1f}%",
            f"Pooled Within-±1 Agreement: {pm.get('within_one_agreement_pct', 0.0):.1f}%",
            f"Pooled Pearson r:           {pm.get('pearson_r', 0.0):.4f}",
            f"Pooled Cohen's κ (unw.):    {pm.get('cohens_kappa_unweighted', 0.0):.4f}",
            f"Pooled Cohen's κ_w (quad.): {pm.get('cohens_kappa_weighted', 0.0):.4f}",
            f"Mean Absolute Error (MAE):  {pm.get('mae', 0.0):.3f} | Mean Signed Bias (Judge - Human): {pm.get('mean_signed_bias', 0.0):+.3f}",
            "-" * 78,
            f"{'Dimension':28} | {'Human':5} | {'Judge':5} | {'Exact%':6} | {'±1%':6} | {'Pearson r':9} | {'κ_w':7}",
            "-" * 78,
        ]
        for dim, m in per_dim.items():
            lines.append(
                f"{dim:28} | {m['human_mean']:5.2f} | {m['judge_mean']:5.2f} | "
                f"{m['exact_agreement_pct']:5.1f}% | {m['within_one_agreement_pct']:5.1f}% | "
                f"{m['pearson_r']:9.4f} | {m['cohens_kappa_weighted']:7.4f}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)

    @classmethod
    def run_verification_pipeline(
        cls,
        runs_dir: str = "runs",
        sample_size: int = 12,
        worksheet_path: str = "runs/human_judge_annotation_sample.json",
        report_output_path: str = "runs/human_judge_agreement_report.json",
    ) -> Dict[str, Any]:
        """
        Loads or generates a sample annotation worksheet, computes human-judge agreement metrics,
        and saves the report JSON to `report_output_path`.
        """
        if os.path.exists(worksheet_path):
            try:
                with open(worksheet_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    samples = data.get("samples", [])
            except Exception:
                samples = []
        else:
            samples = []

        if not samples:
            samples = cls.sample_traces_for_annotation(
                runs_dir=runs_dir,
                sample_size=sample_size,
                output_worksheet_path=worksheet_path,
            )

        agreement = cls.compute_agreement(samples)
        if report_output_path:
            parent = os.path.dirname(report_output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(report_output_path, "w", encoding="utf-8") as f:
                json.dump(agreement, f, indent=2, ensure_ascii=False)

        return agreement


__all__ = [
    "clamp_likert_score",
    "JudgeDimensionScore",
    "LLMJudgeResponseSchema",
    "LLMJudgeReport",
    "PassiveObserver",
    "LLMJudge",
    "HumanJudgeAgreementVerifier",
]
