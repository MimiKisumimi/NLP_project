"""
Batch experiment execution harness and statistical evaluation engine (`experiments.py`).

Orchestrates the 90-run factorial matrix (6 Tasks x 5 Limitation Conditions x 3 Repetitions)
with paired random seeds and failure schedules (`BatchExperimentRunner`), and computes
descriptive statistics, paired t-tests, Cohen's d effect sizes, and communication-to-outcome
Pearson correlations (`StatisticalAnalysisEngine`).
"""

import os
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from scipy import stats

from models import (
    CommunicationIntent,
    LimitationCondition,
)
from config import (
    get_condition_config,
)
from tasks import Task, TaskBenchmarkSuite
from metrics import (
    FailureTrigger,
    FailureScenarioSuite,
)
from observer_judge import (
    PassiveObserver,
    LLMJudge,
)
from environment import Environment
from agent import (
    LLMProvider,
    GeminiProvider,
    RobotAgent,
    get_agent_profile,
    load_env,
)

# Load environment variables from .env if present
load_env()


# ---------------------------------------------------------
# Core Experiment Execution Harness
# ---------------------------------------------------------
@dataclass
class ExperimentMatrixConfig:
    """
    Configuration for the core 90-run experimental evaluation matrix:
    - 6 tasks x 5 constraint conditions x 3 repetitions = 90 executions.
    - Enforces identical LLM model, agent prompts, robot count (3), decoding parameters,
      task definitions, and initial environment states across conditions.
    """

    task_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    conditions: List[LimitationCondition] = field(
        default_factory=lambda: [
            LimitationCondition.REFERENCE,
            LimitationCondition.CAPABILITY_LIMITED,
            LimitationCondition.OBSERVATION_LIMITED,
            LimitationCondition.COMMUNICATION_LIMITED,
            LimitationCondition.COMBINED,
        ]
    )
    repetitions: int = 3
    robot_count: int = 3
    robot_starting_locations: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("Robot_1", "Refrigerator"),
            ("Robot_2", "Cutting table"),
            ("Robot_3", "Stove"),
        ]
    )
    llm_model: str = "gemini-3.5-flash-lite"
    judge_model: Optional[str] = "gemini-3.1-pro-preview"
    decoding_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_predict": 1024,
        }
    )
    max_timesteps_by_difficulty: Dict[str, int] = field(
        default_factory=lambda: {
            "Easy": 26,
            "Medium": 34,
            "Hard": 44,
        }
    )
    base_seed: int = 42
    inject_failures_on_medium_hard: bool = True
    output_dir: str = "runs"

    def total_planned_runs(self) -> int:
        """Returns the total number of episodes in the factorial experiment matrix (`tasks * conditions * repetitions`)."""
        return len(self.task_indices) * len(self.conditions) * self.repetitions

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `ExperimentMatrixConfig` into a JSON-compatible dictionary."""
        return {
            "task_indices": list(self.task_indices),
            "conditions": [c.value for c in self.conditions],
            "repetitions": self.repetitions,
            "total_planned_runs": self.total_planned_runs(),
            "robot_count": self.robot_count,
            "robot_starting_locations": [
                {"robot_id": r_id, "location": str(loc)}
                for r_id, loc in self.robot_starting_locations
            ],
            "llm_model": self.llm_model,
            "judge_model": self.judge_model,
            "decoding_parameters": dict(self.decoding_parameters),
            "max_timesteps_by_difficulty": dict(self.max_timesteps_by_difficulty),
            "base_seed": self.base_seed,
            "inject_failures_on_medium_hard": self.inject_failures_on_medium_hard,
            "output_dir": self.output_dir,
        }


@dataclass
class ExperimentRunRecord:
    """Consolidated record for a single execution in the 90-run matrix."""

    run_id: str
    task_index: int
    task_id: str
    task_name: str
    task_difficulty: str
    condition_key: str
    condition_name: str
    repetition: int
    seed: int
    llm_model: str
    judge_model: str
    robot_count: int
    decoding_parameters: Dict[str, Any]
    trace_filepath: str

    # Task Performance & Coordination Metrics
    task_success: bool
    task_success_rate: float
    # ###Legacy metrics that were not used in the final analysis###
    dag_progress_pct: float
    makespan: int
    valid_action_rate: float
    constraint_violation_rate: float
    total_actions: int
    completed_actions_count: int
    failed_actions_count: int
    total_movements: int
    idle_timesteps: int
    resource_conflicts: int
    dependency_errors: int
    replanning_frequency: int
    recovery_rate: float
    # ###Legacy metrics that were not used in the final analysis###
    workload_gini: float
    workload_distribution: Dict[str, int]

    # Natural-Language Communication Metrics
    number_of_messages: int
    total_communication_tokens: int
    messages_per_successful_task: Optional[float]
    tokens_per_successful_task: Optional[float]
    average_message_length_tokens: float
    # ###Legacy metrics that were not used in the final analysis###
    average_message_length_chars: float
    grounded_message_rate: float
    ungrounded_messages_count: int
    redundant_message_rate: float
    redundant_messages_count: int
    # ###Legacy metrics that were not used in the final analysis###
    comm_volume_to_progress_ratio: float
    communication_function_counts: Dict[str, int]
    communication_function_distribution: Dict[str, float]

    # Independent LLM Judge Semantic Scores
    judge_relevance: int
    judge_clarity: int
    judge_grounding: int
    judge_coordination_utility: int
    judge_redundancy: int
    judge_planning_coherence: int
    judge_failure_explanation_quality: int
    judge_overall_score: float

    # Failure Categories & Injection Summary
    failures_injected_count: int
    failures_detected_count: int
    failures_communicated_count: int
    failures_replanned_count: int
    failures_recovered_count: int
    failure_categories: Dict[str, int]

    # Processing Time, API Latency, Token Usage & Estimated Cost
    processing_time_sec: float
    api_call_count: int
    total_api_latency_ms: float
    mean_api_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_api_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the `ExperimentRunRecord` into a flat dictionary for CSV/JSON export."""
        return {
            "run_id": self.run_id,
            "task_index": self.task_index,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_difficulty": self.task_difficulty,
            "condition_key": self.condition_key,
            "condition_name": self.condition_name,
            "repetition": self.repetition,
            "seed": self.seed,
            "llm_model": self.llm_model,
            "judge_model": self.judge_model,
            "robot_count": self.robot_count,
            "decoding_parameters": self.decoding_parameters,
            "trace_filepath": self.trace_filepath,
            "task_success": self.task_success,
            "task_success_rate": self.task_success_rate,
            "dag_progress_pct": self.dag_progress_pct,
            "makespan": self.makespan,
            "valid_action_rate": self.valid_action_rate,
            "constraint_violation_rate": self.constraint_violation_rate,
            "total_actions": self.total_actions,
            "completed_actions_count": self.completed_actions_count,
            "failed_actions_count": self.failed_actions_count,
            "total_movements": self.total_movements,
            "idle_timesteps": self.idle_timesteps,
            "resource_conflicts": self.resource_conflicts,
            "dependency_errors": self.dependency_errors,
            "replanning_frequency": self.replanning_frequency,
            "recovery_rate": self.recovery_rate,
            "workload_gini": self.workload_gini,
            "workload_distribution": self.workload_distribution,
            "number_of_messages": self.number_of_messages,
            "total_communication_tokens": self.total_communication_tokens,
            "messages_per_successful_task": self.messages_per_successful_task,
            "tokens_per_successful_task": self.tokens_per_successful_task,
            "average_message_length_tokens": self.average_message_length_tokens,
            "average_message_length_chars": self.average_message_length_chars,
            "grounded_message_rate": self.grounded_message_rate,
            "ungrounded_messages_count": self.ungrounded_messages_count,
            "redundant_message_rate": self.redundant_message_rate,
            "redundant_messages_count": self.redundant_messages_count,
            "comm_volume_to_progress_ratio": self.comm_volume_to_progress_ratio,
            "communication_function_counts": self.communication_function_counts,
            "communication_function_distribution": self.communication_function_distribution,
            "judge_relevance": self.judge_relevance,
            "judge_clarity": self.judge_clarity,
            "judge_grounding": self.judge_grounding,
            "judge_coordination_utility": self.judge_coordination_utility,
            "judge_redundancy": self.judge_redundancy,
            "judge_planning_coherence": self.judge_planning_coherence,
            "judge_failure_explanation_quality": self.judge_failure_explanation_quality,
            "judge_overall_score": self.judge_overall_score,
            "failures_injected_count": self.failures_injected_count,
            "failures_detected_count": self.failures_detected_count,
            "failures_communicated_count": self.failures_communicated_count,
            "failures_replanned_count": self.failures_replanned_count,
            "failures_recovered_count": self.failures_recovered_count,
            "failure_categories": self.failure_categories,
            "processing_time_sec": self.processing_time_sec,
            "api_call_count": self.api_call_count,
            "total_api_latency_ms": self.total_api_latency_ms,
            "mean_api_latency_ms": self.mean_api_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_api_tokens": self.total_api_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_trace_dict(
        cls,
        trace_data: Dict[str, Any],
        run_id: Optional[str] = None,
        task_index: Optional[int] = None,
        condition_key: Optional[str] = None,
        condition_name: Optional[str] = None,
        repetition: Optional[int] = None,
        seed: Optional[int] = None,
        trace_filepath: Optional[str] = None,
        default_llm_model: str = "gemini-3.5-flash-lite",
        default_judge_model: str = "gemini-3.1-pro-preview",
        default_decoding_params: Optional[Dict[str, Any]] = None,
        default_robot_count: int = 3,
    ) -> "ExperimentRunRecord":
        """
        Reconstructs an `ExperimentRunRecord` from a saved execution trace dictionary,
        re-extracting objective metrics, communication metrics, LLM-judge scores, failure
        injection recovery telemetry, and API resource telemetry.
        """
        res_run_id = trace_data.get("run_id") or run_id or ""

        # Resolve task details
        task_info = trace_data.get("task") or {}
        res_task_idx = task_index
        if res_task_idx is None:
            t_id_str = str(task_info.get("task_id", ""))
            if t_id_str.startswith("task_") and t_id_str[5:].isdigit():
                res_task_idx = int(t_id_str[5:])
            elif "exp_t" in res_run_id:
                try:
                    res_task_idx = int(res_run_id.split("exp_t")[1].split("_")[0])
                except Exception:
                    res_task_idx = 1
            else:
                res_task_idx = 1

        task_obj = TaskBenchmarkSuite.get_task(res_task_idx)
        task_id = task_info.get("task_id") or (task_obj.task_id if task_obj else f"task_{res_task_idx}")
        task_name = task_info.get("name") or (task_obj.name if task_obj else f"Task {res_task_idx}")
        task_difficulty = task_info.get("difficulty") or (task_obj.difficulty if task_obj else "Medium")

        # Resolve condition details
        cond_info = trace_data.get("condition_config") or {}
        res_cond_key = condition_key or cond_info.get("condition") or "reference"
        res_cond_name = condition_name or cond_info.get("name") or res_cond_key

        # Resolve repetition and deterministic seed
        res_rep = repetition
        if res_rep is None:
            if "_rep" in res_run_id:
                try:
                    res_rep = int(res_run_id.split("_rep")[1])
                except Exception:
                    res_rep = 1
            else:
                res_rep = 1

        res_seed = seed if seed is not None else (42 + (res_task_idx * 100) + res_rep)

        # Telemetry & LLM judge report
        telemetry = trace_data.get("execution_telemetry") or {}
        judge_report = trace_data.get("llm_judge_report") or {}

        llm_model = str(trace_data.get("acting_agent_model") or default_llm_model)
        judge_model = str(judge_report.get("judge_model") or default_judge_model or "")
        robot_count = int(telemetry.get("robot_count") or default_robot_count)
        dec_params = (
            dict(telemetry.get("decoding_parameters"))
            if telemetry.get("decoding_parameters")
            else dict(default_decoding_params or {})
        )

        obj_metrics = trace_data.get("objective_metrics") or {}
        t_metrics = obj_metrics.get("task_performance") or {}
        c_metrics = obj_metrics.get("communication") or {}

        task_success = bool(t_metrics.get("task_success", False))
        task_success_rate = float(
            t_metrics.get("task_success_rate", 1.0 if task_success else 0.0)
        )

        final_state = trace_data.get("final_task_state") or {}
        task_prog = final_state.get("task_progress") or {}
        # ###Legacy metrics that were not used in the final analysis###
        dag_progress_pct = float(
            task_prog.get("percent", t_metrics.get("dag_progress_pct", 100.0 if task_success else 0.0))
        )

        j_scores = (
            judge_report.get("scores")
            or c_metrics.get("llm_judge_scores")
            or {}
        )
        judge_overall = float(
            judge_report.get("overall_average_score")
            or c_metrics.get("llm_judge_overall_score")
            or 3.0
        )

        fm_sum = t_metrics.get("failure_recovery_summary") or {}
        failures_injected_count = int(fm_sum.get("total_failures_injected", 0))

        if "constraint_violations" in trace_data:
            failure_cats = {
                "dependency_precondition_error": 0,
                "resource_lock_conflict": 0,
                "capability_skill_mismatch": 0,
                "navigation_destination_error": 0,
                "communication_budget_exceeded": 0,
                "carrying_capacity_full": 0,
                "injected_environment_or_slip_failure": failures_injected_count,
            }
            for fa in trace_data.get("constraint_violations", []):
                err = str(fa.get("error", "")).lower()
                if "lacks required skill" in err or "not compatible with tool" in err:
                    failure_cats["capability_skill_mismatch"] += 1
                elif "communication limit exceeded" in err or "budget" in err:
                    failure_cats["communication_budget_exceeded"] += 1
                elif "carrying capacity full" in err:
                    failure_cats["carrying_capacity_full"] += 1
                elif "resource conflict" in err or "locked" in err or "occupied" in err:
                    failure_cats["resource_lock_conflict"] += 1
                elif "already at" in err or "destination" in err:
                    failure_cats["navigation_destination_error"] += 1
                elif "injected" in err or "slip" in err:
                    failure_cats["injected_environment_or_slip_failure"] += 1
                else:
                    failure_cats["dependency_precondition_error"] += 1
            if failure_cats["resource_lock_conflict"] == 0:
                failure_cats["resource_lock_conflict"] = (
                    len(trace_data.get("resource_conflicts", []))
                    or int(t_metrics.get("resource_conflicts", 0))
                )
        elif "failure_categories" in trace_data:
            failure_cats = dict(trace_data["failure_categories"])
        else:
            failure_cats = {
                "dependency_precondition_error": 0,
                "resource_lock_conflict": (
                    len(trace_data.get("resource_conflicts", []))
                    or int(t_metrics.get("resource_conflicts", 0))
                ),
                "capability_skill_mismatch": 0,
                "navigation_destination_error": 0,
                "communication_budget_exceeded": 0,
                "carrying_capacity_full": 0,
                "injected_environment_or_slip_failure": failures_injected_count,
            }

        return cls(
            run_id=res_run_id,
            task_index=res_task_idx,
            task_id=task_id,
            task_name=task_name,
            task_difficulty=task_difficulty,
            condition_key=res_cond_key,
            condition_name=res_cond_name,
            repetition=res_rep,
            seed=res_seed,
            llm_model=llm_model,
            judge_model=judge_model,
            robot_count=robot_count,
            decoding_parameters=dec_params,
            trace_filepath=trace_filepath or "",
            task_success=task_success,
            task_success_rate=task_success_rate,
            dag_progress_pct=dag_progress_pct,
            makespan=int(t_metrics.get("makespan", 0)),
            valid_action_rate=float(t_metrics.get("valid_action_rate", 0.0)),
            constraint_violation_rate=float(t_metrics.get("constraint_violation_rate", 0.0)),
            total_actions=int(t_metrics.get("total_actions", 0)),
            completed_actions_count=int(t_metrics.get("completed_actions_count", 0)),
            failed_actions_count=int(t_metrics.get("failed_actions_count", 0)),
            total_movements=int(t_metrics.get("total_movements", 0)),
            idle_timesteps=int(t_metrics.get("idle_timesteps", 0)),
            resource_conflicts=int(t_metrics.get("resource_conflicts", 0)),
            dependency_errors=int(t_metrics.get("dependency_errors", 0)),
            replanning_frequency=int(t_metrics.get("replanning_frequency", 0)),
            recovery_rate=float(t_metrics.get("recovery_rate", 1.0)),
            workload_gini=float(t_metrics.get("workload_gini", 0.0)),
            workload_distribution=dict(t_metrics.get("workload_distribution") or {}),
            number_of_messages=int(c_metrics.get("number_of_messages", 0)),
            total_communication_tokens=int(c_metrics.get("total_communication_tokens", 0)),
            messages_per_successful_task=(
                float(c_metrics["messages_per_successful_task"])
                if c_metrics.get("messages_per_successful_task") is not None
                else None
            ),
            tokens_per_successful_task=(
                float(c_metrics["tokens_per_successful_task"])
                if c_metrics.get("tokens_per_successful_task") is not None
                else None
            ),
            average_message_length_tokens=float(c_metrics.get("average_message_length_tokens", 0.0)),
            average_message_length_chars=float(c_metrics.get("average_message_length_chars", 0.0)),
            grounded_message_rate=float(c_metrics.get("grounded_message_rate", 1.0)),
            ungrounded_messages_count=int(c_metrics.get("ungrounded_messages_count", 0)),
            redundant_message_rate=float(c_metrics.get("redundant_message_rate", 0.0)),
            redundant_messages_count=int(c_metrics.get("redundant_messages_count", 0)),
            comm_volume_to_progress_ratio=float(c_metrics.get("comm_volume_to_progress_ratio", 0.0)),
            communication_function_counts=dict(c_metrics.get("communication_function_counts") or {}),
            communication_function_distribution=dict(
                c_metrics.get("communication_function_distribution") or {}
            ),
            judge_relevance=int(j_scores.get("relevance", 3)),
            judge_clarity=int(j_scores.get("clarity", 3)),
            judge_grounding=int(j_scores.get("grounding", 3)),
            judge_coordination_utility=int(j_scores.get("coordination_utility", 3)),
            judge_redundancy=int(j_scores.get("redundancy", 3)),
            judge_planning_coherence=int(j_scores.get("planning_coherence", 3)),
            judge_failure_explanation_quality=int(
                j_scores.get("failure_explanation_quality", 3)
            ),
            judge_overall_score=judge_overall,
            failures_injected_count=failures_injected_count,
            failures_detected_count=int(fm_sum.get("failures_detected", 0)),
            failures_communicated_count=int(fm_sum.get("failures_communicated", 0)),
            failures_replanned_count=int(fm_sum.get("plans_revised", 0)),
            failures_recovered_count=int(fm_sum.get("failures_recovered", 0)),
            failure_categories=failure_cats,
            processing_time_sec=float(telemetry.get("processing_time_sec", 0.0)),
            api_call_count=int(telemetry.get("api_call_count", 0)),
            total_api_latency_ms=float(telemetry.get("total_api_latency_ms", 0.0)),
            mean_api_latency_ms=float(telemetry.get("mean_api_latency_ms", 0.0)),
            prompt_tokens=int(telemetry.get("prompt_tokens", 0)),
            completion_tokens=int(telemetry.get("completion_tokens", 0)),
            total_api_tokens=int(telemetry.get("total_api_tokens", 0)),
            estimated_cost_usd=float(telemetry.get("estimated_cost_usd", 0.0)),
        )


class BatchExperimentRunner:
    """
    Executes the 6 Tasks x 5 Conditions x 3 Repetitions = 90 Runs evaluation matrix.
    Ensures identical model, prompts, robot count, decoding parameters, task definitions,
    initial states, and seed-controlled failure triggers across all conditions.
    """

    def __init__(
        self,
        config: Optional[ExperimentMatrixConfig] = None,
        verbose: bool = False,
    ):
        """Initializes the `BatchExperimentRunner` with an `ExperimentMatrixConfig` and `LLMJudge` instance."""
        self.config = config or ExperimentMatrixConfig()
        self.verbose = verbose
        self.judge = LLMJudge(
            judge_model=self.config.judge_model,
            acting_agent_model=self.config.llm_model,
        )

    @staticmethod
    def get_predetermined_failures_for_task(
        task: Task, repetition: int, seed: int, enabled: bool = True
    ) -> List[FailureTrigger]:
        """
        Returns predetermined, seed-controlled failure scenarios for Medium and Hard tasks
        so that conditions remain comparable.
        Because triggers depend ONLY on (task.task_id, repetition, seed), every limitation condition
        faces the exact same failure schedule for a given task and repetition.
        """
        if not enabled or task.difficulty == "Easy":
            return []

        if task.difficulty == "Medium":
            # Deterministically select one controlled failure keyed by task & repetition
            if repetition == 1:
                return FailureScenarioSuite.get_scenario_by_option(
                    2, seed=seed
                )  # Tool Temporarily Unavailable
            elif repetition == 2:
                return FailureScenarioSuite.get_scenario_by_option(
                    4, seed=seed
                )  # Action Execution Slip
            else:
                return FailureScenarioSuite.get_scenario_by_option(
                    6, seed=seed
                )  # Status Message Delayed

        # Hard tasks: predetermined failure scenarios testing resource contention, delivery, and tool locks
        if repetition == 1:
            return FailureScenarioSuite.get_scenario_by_option(
                3, seed=seed
            )  # Workstation Occupied / Blocked
        elif repetition == 2:
            return FailureScenarioSuite.get_scenario_by_option(
                5, seed=seed
            )  # Misplaced Delivery
        else:
            return FailureScenarioSuite.get_scenario_by_option(
                7, seed=seed
            )  # Combined Multi-Failure Suite

    @staticmethod
    def categorize_failures(env: Environment) -> Dict[str, int]:
        """
        Categorizes all rejected actions, resource conflicts, and injected failures into standard
        failure categories for reporting across conditions.
        """
        categories = {
            "dependency_precondition_error": 0,
            "resource_lock_conflict": 0,
            "capability_skill_mismatch": 0,
            "navigation_destination_error": 0,
            "communication_budget_exceeded": 0,
            "carrying_capacity_full": 0,
            "injected_environment_or_slip_failure": (
                len(getattr(env.failure_manager, "events", []))
                if getattr(env, "failure_manager", None)
                else 0
            ),
        }

        for fa in getattr(env, "failed_actions", []):
            err = str(fa.get("error", "")).lower()
            if "lacks required skill" in err or "not compatible with tool" in err:
                categories["capability_skill_mismatch"] += 1
            elif "communication limit exceeded" in err or "budget" in err:
                categories["communication_budget_exceeded"] += 1
            elif "carrying capacity full" in err:
                categories["carrying_capacity_full"] += 1
            elif "resource conflict" in err or "locked" in err or "occupied" in err:
                categories["resource_lock_conflict"] += 1
            elif "already at" in err or "destination" in err:
                categories["navigation_destination_error"] += 1
            elif "injected" in err or "slip" in err:
                categories["injected_environment_or_slip_failure"] += 1
            else:
                categories["dependency_precondition_error"] += 1

        if categories["resource_lock_conflict"] == 0:
            categories["resource_lock_conflict"] = len(
                getattr(env, "resource_conflicts", [])
            )

        return categories

    def run_single_episode(
        self,
        task_index: int,
        condition: LimitationCondition,
        repetition: int,
    ) -> ExperimentRunRecord:
        """
        Executes a single (task, condition, repetition) episode with identical initial state,
        robot count, model, and failure seed across conditions.
        """
        import sys
        import io

        # Fresh task instance so initial items and DAG nodes are clean and identical across conditions
        task = TaskBenchmarkSuite.get_task(task_index)
        if task is None:
            task = TaskBenchmarkSuite.task_1_prepare_salad()

        cond_cfg = get_condition_config(condition)
        # Seed depends ONLY on (task_index, repetition) so paired runs across the 5 conditions share the exact seed!
        paired_seed = self.config.base_seed + (task_index * 100) + repetition

        failure_triggers = self.get_predetermined_failures_for_task(
            task=task,
            repetition=repetition,
            seed=paired_seed,
            enabled=self.config.inject_failures_on_medium_hard,
        )

        run_id = f"exp_t{task_index}_{cond_cfg.condition.value}_rep{repetition}"
        observer = PassiveObserver(run_id=run_id, output_dir=self.config.output_dir)

        # Suppress verbose per-tick console prints during batch matrix runs unless verbose=True
        saved_stdout = sys.stdout
        if not self.verbose:
            sys.stdout = io.StringIO()

        try:
            env = Environment(
                task=task,
                condition=cond_cfg,
                failure_triggers=failure_triggers,
                failure_seed=paired_seed,
                observer=observer,
                run_id=run_id,
            )

            # Initialize the 3 robots with identical starting positions and decoding parameters
            dec_params = self.config.decoding_parameters

            for r_id, start_pos in self.config.robot_starting_locations[
                : self.config.robot_count
            ]:
                prof = get_agent_profile(r_id, cond_cfg)
                provider: LLMProvider = GeminiProvider(
                    model=self.config.llm_model,
                    temperature=float(dec_params.get("temperature", 0.2)),
                )

                robot = RobotAgent(
                    r_id,
                    skills=prof["skills"],
                    carrying_capacity=prof["carrying_capacity"],
                    tool_compatibility=prof["tool_compatibility"],
                    role=prof["role"],
                    description=prof["description"],
                    llm_provider=provider,
                    memory_limit=prof["memory_limit"],
                    max_messages=prof["max_messages"],
                    max_comm_tokens=prof["max_comm_tokens"],
                )
                env.add_agent(robot, starting_location=start_pos)

            # Broadcast task at t=0 and run simulation loop
            env.broadcast_task_at_start()
            max_steps = 45

            for t in range(1, max_steps + 1):
                env.timestep(t)
                if env.is_task_complete():
                    break

            trace_filepath = os.path.join(self.config.output_dir, f"{run_id}.json")
            t_metrics, c_metrics, j_report, saved_path = env.evaluate_and_export_trace(
                judge=self.judge,
                filepath=trace_filepath,
            )
            failure_cats = self.categorize_failures(env)
            telemetry = env.observer.execution_telemetry
            # ###Legacy metrics that were not used in the final analysis###
            dag_prog_pct = (
                float(env.current_task.get_progress().get("percent", 0.0))
                if env.current_task
                else 0.0
            )
            fm_sum = t_metrics.failure_recovery_summary or {}

        finally:
            if not self.verbose:
                sys.stdout = saved_stdout

        j_scores = j_report.get_scores_dict()
        return ExperimentRunRecord(
            run_id=run_id,
            task_index=task_index,
            task_id=task.task_id,
            task_name=task.name,
            task_difficulty=task.difficulty,
            condition_key=cond_cfg.condition.value,
            condition_name=cond_cfg.name,
            repetition=repetition,
            seed=paired_seed,
            llm_model=self.config.llm_model,
            judge_model=j_report.judge_model,
            robot_count=self.config.robot_count,
            decoding_parameters=dict(self.config.decoding_parameters),
            trace_filepath=saved_path,
            task_success=t_metrics.task_success,
            task_success_rate=t_metrics.task_success_rate,
            dag_progress_pct=dag_prog_pct,
            makespan=t_metrics.makespan,
            valid_action_rate=t_metrics.valid_action_rate,
            constraint_violation_rate=t_metrics.constraint_violation_rate,
            total_actions=t_metrics.total_actions,
            completed_actions_count=t_metrics.completed_actions_count,
            failed_actions_count=t_metrics.failed_actions_count,
            total_movements=t_metrics.total_movements,
            idle_timesteps=t_metrics.idle_timesteps,
            resource_conflicts=t_metrics.resource_conflicts,
            dependency_errors=t_metrics.dependency_errors,
            replanning_frequency=t_metrics.replanning_frequency,
            recovery_rate=t_metrics.recovery_rate,
            workload_gini=t_metrics.workload_gini,
            workload_distribution=dict(t_metrics.workload_distribution),
            number_of_messages=c_metrics.number_of_messages,
            total_communication_tokens=c_metrics.total_communication_tokens,
            messages_per_successful_task=c_metrics.messages_per_successful_task,
            tokens_per_successful_task=c_metrics.tokens_per_successful_task,
            average_message_length_tokens=c_metrics.average_message_length_tokens,
            average_message_length_chars=c_metrics.average_message_length_chars,
            grounded_message_rate=c_metrics.grounded_message_rate,
            ungrounded_messages_count=c_metrics.ungrounded_messages_count,
            redundant_message_rate=c_metrics.redundant_message_rate,
            redundant_messages_count=c_metrics.redundant_messages_count,
            comm_volume_to_progress_ratio=c_metrics.comm_volume_to_progress_ratio,
            communication_function_counts=dict(c_metrics.communication_function_counts),
            communication_function_distribution=dict(
                c_metrics.communication_function_distribution
            ),
            judge_relevance=j_scores.get("relevance", 3),
            judge_clarity=j_scores.get("clarity", 3),
            judge_grounding=j_scores.get("grounding", 3),
            judge_coordination_utility=j_scores.get("coordination_utility", 3),
            judge_redundancy=j_scores.get("redundancy", 3),
            judge_planning_coherence=j_scores.get("planning_coherence", 3),
            judge_failure_explanation_quality=j_scores.get(
                "failure_explanation_quality", 3
            ),
            judge_overall_score=j_report.overall_average_score,
            failures_injected_count=int(fm_sum.get("total_failures_injected", 0)),
            failures_detected_count=int(fm_sum.get("failures_detected", 0)),
            failures_communicated_count=int(fm_sum.get("failures_communicated", 0)),
            failures_replanned_count=int(fm_sum.get("plans_revised", 0)),
            failures_recovered_count=int(fm_sum.get("failures_recovered", 0)),
            failure_categories=failure_cats,
            processing_time_sec=float(telemetry.get("processing_time_sec", 0.0)),
            api_call_count=int(telemetry.get("api_call_count", 0)),
            total_api_latency_ms=float(telemetry.get("total_api_latency_ms", 0.0)),
            mean_api_latency_ms=float(telemetry.get("mean_api_latency_ms", 0.0)),
            prompt_tokens=int(telemetry.get("prompt_tokens", 0)),
            completion_tokens=int(telemetry.get("completion_tokens", 0)),
            total_api_tokens=int(telemetry.get("total_api_tokens", 0)),
            estimated_cost_usd=float(telemetry.get("estimated_cost_usd", 0.0)),
        )

    def _reconstruct_record_from_trace(
        self,
        trace_data: Dict[str, Any],
        run_id: str,
        trace_filepath: str,
        task_index: int,
        cond_cfg: Any,
        repetition: int,
    ) -> ExperimentRunRecord:
        """
        Reconstructs an ExperimentRunRecord from a saved trace JSON file's objective_metrics
        and execution_telemetry when resuming an interrupted experiment.
        """
        paired_seed = self.config.base_seed + (task_index * 100) + repetition
        return ExperimentRunRecord.from_trace_dict(
            trace_data=trace_data,
            run_id=run_id,
            task_index=task_index,
            condition_key=cond_cfg.condition.value,
            condition_name=cond_cfg.name,
            repetition=repetition,
            seed=paired_seed,
            trace_filepath=trace_filepath,
            default_llm_model=self.config.llm_model,
            default_judge_model=self.config.judge_model or "",
            default_decoding_params=self.config.decoding_parameters,
            default_robot_count=self.config.robot_count,
        )

    def run_all(
        self, progress_callback: Optional[Any] = None
    ) -> List[ExperimentRunRecord]:
        """
        Executes all runs in the configured matrix, reusing existing `.json` trace files
        in `output_dir` to support resuming interrupted experiments.
        """
        records: List[ExperimentRunRecord] = []
        total = self.config.total_planned_runs()
        idx = 0

        for task_idx in self.config.task_indices:
            for cond in self.config.conditions:
                for rep in range(1, self.config.repetitions + 1):
                    idx += 1
                    cond_cfg = get_condition_config(cond)
                    run_id = f"exp_t{task_idx}_{cond_cfg.condition.value}_rep{rep}"
                    trace_filepath = os.path.join(self.config.output_dir, f"{run_id}.json")

                    if os.path.exists(trace_filepath):
                        with open(trace_filepath, "r", encoding="utf-8") as f:
                            trace_data = json.load(f)
                        rec = self._reconstruct_record_from_trace(
                            trace_data=trace_data,
                            run_id=run_id,
                            trace_filepath=trace_filepath,
                            task_index=task_idx,
                            cond_cfg=cond_cfg,
                            repetition=rep,
                        )
                    else:
                        rec = self.run_single_episode(
                            task_index=task_idx, condition=cond, repetition=rep
                        )

                    records.append(rec)
                    if progress_callback:
                        progress_callback(idx, total, rec)
                    else:
                        status_icon = (
                            "PASS"
                            if rec.task_success
                            else f"{rec.dag_progress_pct:.0f}%"
                        )
                        print(
                            f"[{idx:02d}/{total:02d}] {rec.run_id:34} | "
                            f"Task {rec.task_index} ({rec.task_difficulty:6}) | {rec.condition_name:21} | "
                            f"Rep {rec.repetition} | Status: {status_icon:4} | "
                            f"Makespan: {rec.makespan:2d}s | Msgs: {rec.number_of_messages:2d} | "
                            f"Judge: {rec.judge_overall_score:.2f}/5"
                        )
        return records


# ---------------------------------------------------------
# Statistical Analysis & Comparisons Engine
# ---------------------------------------------------------
class StatisticalAnalysisEngine:
    """
    Aggregates per-condition descriptive statistics (Mean, SD, 95% CI), paired comparisons
    against the Reference baseline (paired t-test and Cohen's d), failure/telemetry summaries,
    and communication-to-outcome correlations, exporting CSV, JSON, and Markdown reports.
    """

    KEY_METRICS: List[Tuple[str, str]] = [
        ("task_success_rate", "Task Success Rate"),
        # ###Legacy metrics that were not used in the final analysis###
        ("dag_progress_pct", "DAG Completion (%)"),
        ("makespan", "Makespan (s)"),
        ("valid_action_rate", "Valid Action Rate"),
        ("constraint_violation_rate", "Constraint Violation Rate"),
        ("total_actions", "Total Actions"),
        ("total_movements", "Total Movements"),
        ("idle_timesteps", "Idle Timesteps (s)"),
        ("resource_conflicts", "Resource Conflicts"),
        ("dependency_errors", "Dependency Errors"),
        ("replanning_frequency", "Replanning Frequency"),
        ("recovery_rate", "Post-Failure Recovery Rate"),
        # ###Legacy metrics that were not used in the final analysis###
        ("workload_gini", "Workload Gini Coefficient"),
        ("number_of_messages", "Number of Peer Messages"),
        ("total_communication_tokens", "Total Comm Tokens"),
        ("average_message_length_tokens", "Avg Message Length (Tokens)"),
        ("grounded_message_rate", "Grounded Message Rate"),
        ("redundant_message_rate", "Redundant Message Rate"),
        ("judge_relevance", "LLM Judge Relevance (1-5)"),
        ("judge_clarity", "LLM Judge Clarity (1-5)"),
        ("judge_grounding", "LLM Judge Grounding (1-5)"),
        ("judge_coordination_utility", "LLM Judge Coordination Utility (1-5)"),
        ("judge_redundancy", "LLM Judge Non-Redundancy (1-5)"),
        ("judge_planning_coherence", "LLM Judge Planning Coherence (1-5)"),
        ("judge_failure_explanation_quality", "LLM Judge Failure Explanation (1-5)"),
        ("judge_overall_score", "LLM Judge Overall Score (1-5)"),
        ("processing_time_sec", "Processing Time (s)"),
        ("mean_api_latency_ms", "Mean API Latency (ms)"),
        ("total_api_tokens", "Total API Tokens"),
        ("estimated_cost_usd", "Estimated API Cost (USD)"),
    ]

    @classmethod
    def mean_sd_ci95(cls, values: List[float]) -> Dict[str, float]:
        """Computes Mean, Sample Standard Deviation (SD), and 95% Confidence Interval [ci_low, ci_high]."""
        n = len(values)
        if n == 0:
            return {
                "n": 0,
                "mean": 0.0,
                "sd": 0.0,
                "se": 0.0,
                "ci95_half_width": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
            }
        arr = np.asarray(values, dtype=float)
        mean_val = float(np.mean(arr))
        if n == 1:
            return {
                "n": 1,
                "mean": round(mean_val, 4),
                "sd": 0.0,
                "se": 0.0,
                "ci95_half_width": 0.0,
                "ci95_low": round(mean_val, 4),
                "ci95_high": round(mean_val, 4),
            }
        sd_val = float(np.std(arr, ddof=1))
        se_val = float(stats.sem(arr)) if sd_val > 0 else 0.0
        half_w = float(stats.t.ppf(0.975, df=n - 1)) * se_val
        return {
            "n": n,
            "mean": round(mean_val, 4),
            "sd": round(sd_val, 4),
            "se": round(se_val, 4),
            "ci95_half_width": round(half_w, 4),
            "ci95_low": round(mean_val - half_w, 4),
            "ci95_high": round(mean_val + half_w, 4),
        }

    @staticmethod
    def _safe_p(p_val: Any) -> float:
        """Converts a scipy p-value to a rounded float, mapping NaN to 1.0."""
        p = float(p_val)
        return 1.0 if math.isnan(p) else round(p, 6)

    @classmethod
    def paired_t_test(
        cls, ref_values: List[float], cond_values: List[float]
    ) -> Dict[str, Any]:
        """
        Computes a two-tailed paired t-test (`scipy.stats.ttest_rel`) between `cond_values`
        and `ref_values` matched by (task_index, repetition).
        """
        n = min(len(ref_values), len(cond_values))
        if n < 2:
            return {
                "n": n,
                "mean_diff": 0.0,
                "sd_diff": 0.0,
                "t_stat": 0.0,
                "df": max(0, n - 1),
                "p_value": 1.0,
                "significant_p05": False,
            }

        ref = np.asarray(ref_values[:n], dtype=float)
        cond = np.asarray(cond_values[:n], dtype=float)
        diffs = cond - ref
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))

        if sd_d < 1e-12:
            # Zero-variance differences: t-test undefined
            return {
                "n": n,
                "mean_diff": round(mean_d, 4),
                "sd_diff": 0.0,
                "t_stat": 0.0,
                "df": n - 1,
                "p_value": 1.0,
                "significant_p05": False,
            }

        res = stats.ttest_rel(cond, ref)
        p_val = cls._safe_p(res.pvalue)
        return {
            "n": n,
            "mean_diff": round(mean_d, 4),
            "sd_diff": round(sd_d, 4),
            "t_stat": round(float(res.statistic), 4),
            "df": n - 1,
            "p_value": p_val,
            "significant_p05": bool(p_val < 0.05),
        }

    # ###Legacy metrics that were not used in the final analysis###
    @classmethod
    def wilcoxon_signed_rank(
        cls, ref_values: List[float], cond_values: List[float]
    ) -> Dict[str, Any]:
        """
        Computes the non-parametric Wilcoxon signed-rank test (`scipy.stats.wilcoxon`)
        on paired differences (cond_values[i] - ref_values[i]).
        """
        n_total = min(len(ref_values), len(cond_values))
        diffs = np.asarray(cond_values[:n_total], dtype=float) - np.asarray(
            ref_values[:n_total], dtype=float
        )
        non_zero = diffs[np.abs(diffs) > 1e-12]
        n_eff = int(non_zero.size)
        if n_eff == 0:
            return {
                "n_effective": 0,
                "w_stat": 0.0,
                "z_stat": 0.0,
                "p_value": 1.0,
                "significant_p05": False,
            }

        ranks = stats.rankdata(np.abs(non_zero))
        w_pos = float(np.sum(ranks[non_zero > 0]))
        w_neg = float(np.sum(ranks[non_zero < 0]))

        try:
            res = stats.wilcoxon(non_zero, zero_method="wilcox", correction=True)
            w_stat = float(res.statistic)
            p_val = cls._safe_p(res.pvalue)
        except ValueError:
            w_stat, p_val = min(w_pos, w_neg), 1.0

        # Normal-approximation z for reporting, derived from the two-tailed p-value
        z_stat = float(stats.norm.isf(p_val / 2.0)) if 0.0 < p_val < 1.0 else 0.0
        if w_pos < w_neg:
            z_stat = -z_stat

        return {
            "n_effective": n_eff,
            "w_stat": round(w_stat, 3),
            "w_plus": round(w_pos, 3),
            "w_minus": round(w_neg, 3),
            "z_stat": round(z_stat, 4),
            "p_value": p_val,
            "significant_p05": bool(p_val < 0.05),
        }

    @staticmethod
    def cohens_d(ref_values: List[float], cond_values: List[float]) -> Dict[str, Any]:
        """
        Computes Cohen's d effect size (both pooled independent SD and paired difference SD_z)
        with standard magnitude interpretation (negligible, small, medium, large).
        """
        n1, n2 = len(ref_values), len(cond_values)
        if n1 < 2 or n2 < 2:
            return {
                "cohens_d_pooled": 0.0,
                "cohens_d_paired": 0.0,
                "magnitude": "negligible",
            }

        ref = np.asarray(ref_values, dtype=float)
        cond = np.asarray(cond_values, dtype=float)
        pooled_var = (
            (n1 - 1) * np.var(ref, ddof=1) + (n2 - 1) * np.var(cond, ddof=1)
        ) / float(n1 + n2 - 2)
        pooled_sd = float(np.sqrt(max(0.0, pooled_var)))
        mean_gap = float(np.mean(cond) - np.mean(ref))
        d_pooled = (mean_gap / pooled_sd) if pooled_sd > 1e-12 else 0.0

        # ###Legacy metrics that were not used in the final analysis###
        # Paired Cohen's d_z
        n_pair = min(n1, n2)
        diffs = cond[:n_pair] - ref[:n_pair]
        sd_d = float(np.std(diffs, ddof=1))
        d_paired = (float(np.mean(diffs)) / sd_d) if sd_d > 1e-12 else d_pooled

        abs_d = abs(d_pooled)
        if abs_d < 0.2:
            mag = "negligible"
        elif abs_d < 0.5:
            mag = "small"
        elif abs_d < 0.8:
            mag = "medium"
        else:
            mag = "large"

        return {
            "cohens_d_pooled": round(d_pooled, 4),
            "cohens_d_paired": round(d_paired, 4),
            "magnitude": mag,
        }

    @classmethod
    def _pearson_correlation(cls, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Computes Pearson correlation coefficient r and its two-tailed p-value (`scipy.stats.pearsonr`)."""
        n = min(len(x), len(y))
        if n < 3:
            return 0.0, 1.0
        xa = np.asarray(x[:n], dtype=float)
        ya = np.asarray(y[:n], dtype=float)
        if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
            return 0.0, 1.0
        r, p = stats.pearsonr(xa, ya)
        return round(float(r), 4), cls._safe_p(p)

    @classmethod
    def analyze_communication_predictors(
        cls, records: List[ExperimentRunRecord]
    ) -> Dict[str, Any]:
        """
        Examines whether communication characteristics such as grounding, relevance, redundancy,
        clarity, coordination utility, and message volume predict task success or post-failure
        recovery.
        """
        comm_features = [
            ("grounded_message_rate", "Grounded Message Rate (Simulator)"),
            ("redundant_message_rate", "Redundant Message Rate (Simulator)"),
            ("judge_relevance", "LLM Judge Relevance (1-5)"),
            ("judge_clarity", "LLM Judge Clarity (1-5)"),
            ("judge_grounding", "LLM Judge Grounding (1-5)"),
            ("judge_coordination_utility", "LLM Judge Coordination Utility (1-5)"),
            ("judge_redundancy", "LLM Judge Non-Redundancy (1-5)"),
            ("judge_planning_coherence", "LLM Judge Planning Coherence (1-5)"),
            (
                "judge_failure_explanation_quality",
                "LLM Judge Failure Explanation (1-5)",
            ),
            ("judge_overall_score", "LLM Judge Overall Score (1-5)"),
            ("number_of_messages", "Number of Peer Messages"),
            ("total_communication_tokens", "Total Communication Tokens"),
        ]

        y_success = [float(r.task_success_rate) for r in records]
        # ###Legacy metrics that were not used in the final analysis###
        y_progress = [float(r.dag_progress_pct) for r in records]
        y_recovery = [float(r.recovery_rate) for r in records]

        correlations: Dict[str, Dict[str, Any]] = {}
        for feat_key, feat_label in comm_features:
            x_vals = [float(getattr(r, feat_key, 0.0) or 0.0) for r in records]
            r_succ, p_succ = cls._pearson_correlation(x_vals, y_success)
            # ###Legacy metrics that were not used in the final analysis###
            r_prog, p_prog = cls._pearson_correlation(x_vals, y_progress)
            r_rec, p_rec = cls._pearson_correlation(x_vals, y_recovery)

            succ_group = [
                float(getattr(r, feat_key, 0.0) or 0.0)
                for r in records
                if r.task_success
            ]
            fail_group = [
                float(getattr(r, feat_key, 0.0) or 0.0)
                for r in records
                if not r.task_success
            ]

            correlations[feat_key] = {
                "label": feat_label,
                "predicts_task_success_r": r_succ,
                "predicts_task_success_p": p_succ,
                "predicts_dag_progress_r": r_prog,
                "predicts_dag_progress_p": p_prog,
                "predicts_recovery_rate_r": r_rec,
                "predicts_recovery_rate_p": p_rec,
                "mean_in_successful_runs": (
                    round(sum(succ_group) / len(succ_group), 3) if succ_group else 0.0
                ),
                "mean_in_unsuccessful_runs": (
                    round(sum(fail_group) / len(fail_group), 3) if fail_group else 0.0
                ),
            }

        return correlations

    @classmethod
    def _group_records_by_condition(
        cls,
        records: List[ExperimentRunRecord],
    ) -> Tuple[Dict[str, List[ExperimentRunRecord]], List[str]]:
        """Groups run records by condition_key and returns ordered present condition keys."""
        by_condition: Dict[str, List[ExperimentRunRecord]] = {}
        for rec in records:
            by_condition.setdefault(rec.condition_key, []).append(rec)

        condition_order = [
            LimitationCondition.REFERENCE.value,
            LimitationCondition.CAPABILITY_LIMITED.value,
            LimitationCondition.OBSERVATION_LIMITED.value,
            LimitationCondition.COMMUNICATION_LIMITED.value,
            LimitationCondition.COMBINED.value,
        ]
        present_conditions = [c for c in condition_order if c in by_condition] + [
            c for c in by_condition if c not in condition_order
        ]
        return by_condition, present_conditions

    @classmethod
    def _summarize_single_condition(
        cls,
        cond_key: str,
        cond_recs: List[ExperimentRunRecord],
    ) -> Dict[str, Any]:
        """Computes descriptive statistics (Mean, SD, 95% CI), failure counts, speech acts, and telemetry for one condition."""
        cond_name = cond_recs[0].condition_name if cond_recs else cond_key
        metrics_agg: Dict[str, Dict[str, float]] = {}

        for m_key, _ in cls.KEY_METRICS:
            vals = [float(getattr(r, m_key, 0.0) or 0.0) for r in cond_recs]
            metrics_agg[m_key] = cls.mean_sd_ci95(vals)

        succ_msgs = [
            float(r.messages_per_successful_task)
            for r in cond_recs
            if r.messages_per_successful_task is not None
        ]
        succ_toks = [
            float(r.tokens_per_successful_task)
            for r in cond_recs
            if r.tokens_per_successful_task is not None
        ]
        metrics_agg["messages_per_successful_task"] = cls.mean_sd_ci95(succ_msgs)
        metrics_agg["tokens_per_successful_task"] = cls.mean_sd_ci95(succ_toks)

        agg_failure_cats: Dict[str, int] = {}
        for r in cond_recs:
            for f_cat, f_cnt in r.failure_categories.items():
                agg_failure_cats[f_cat] = agg_failure_cats.get(f_cat, 0) + int(f_cnt)

        agg_speech_acts: Dict[str, int] = {
            intent.value: 0 for intent in CommunicationIntent
        }
        for r in cond_recs:
            for act_k, act_cnt in r.communication_function_counts.items():
                agg_speech_acts[act_k] = agg_speech_acts.get(act_k, 0) + int(act_cnt)
        total_cond_msgs = sum(agg_speech_acts.values())
        speech_act_pct = {
            k: (
                round((v / float(total_cond_msgs)) * 100.0, 1)
                if total_cond_msgs > 0
                else 0.0
            )
            for k, v in agg_speech_acts.items()
        }

        return {
            "condition_key": cond_key,
            "condition_name": cond_name,
            "num_runs": len(cond_recs),
            "successful_runs": sum(1 for r in cond_recs if r.task_success),
            "metrics": metrics_agg,
            "failure_categories_total": agg_failure_cats,
            "speech_act_counts_total": agg_speech_acts,
            "speech_act_distribution_pct": speech_act_pct,
            "telemetry_totals": {
                "total_processing_time_sec": round(
                    sum(r.processing_time_sec for r in cond_recs), 3
                ),
                "total_api_calls": sum(r.api_call_count for r in cond_recs),
                "mean_api_latency_ms": metrics_agg["mean_api_latency_ms"]["mean"],
                "total_prompt_tokens": sum(r.prompt_tokens for r in cond_recs),
                "total_completion_tokens": sum(r.completion_tokens for r in cond_recs),
                "total_api_tokens": sum(r.total_api_tokens for r in cond_recs),
                "total_estimated_cost_usd": round(
                    sum(r.estimated_cost_usd for r in cond_recs), 6
                ),
            },
        }

    @classmethod
    def _compute_paired_comparisons(
        cls,
        by_condition: Dict[str, List[ExperimentRunRecord]],
        present_conditions: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Runs paired t-tests, Wilcoxon signed-rank tests, and Cohen's d against the Reference condition."""
        ref_key = LimitationCondition.REFERENCE.value
        paired_comparisons: Dict[str, Dict[str, Any]] = {}
        if ref_key not in by_condition:
            return paired_comparisons

        ref_lookup: Dict[Tuple[int, int], ExperimentRunRecord] = {
            (r.task_index, r.repetition): r for r in by_condition[ref_key]
        }

        for cond_key in present_conditions:
            if cond_key == ref_key:
                continue
            cond_lookup: Dict[Tuple[int, int], ExperimentRunRecord] = {
                (r.task_index, r.repetition): r for r in by_condition[cond_key]
            }
            shared_keys = sorted(set(ref_lookup.keys()) & set(cond_lookup.keys()))
            cond_name = (
                by_condition[cond_key][0].condition_name
                if by_condition[cond_key]
                else cond_key
            )

            metric_tests: Dict[str, Dict[str, Any]] = {}
            for m_key, m_label in cls.KEY_METRICS:
                ref_vec = [
                    float(getattr(ref_lookup[k], m_key, 0.0) or 0.0)
                    for k in shared_keys
                ]
                cond_vec = [
                    float(getattr(cond_lookup[k], m_key, 0.0) or 0.0)
                    for k in shared_keys
                ]

                t_res = cls.paired_t_test(ref_vec, cond_vec)
                # ###Legacy metrics that were not used in the final analysis###
                w_res = cls.wilcoxon_signed_rank(ref_vec, cond_vec)
                d_res = cls.cohens_d(ref_vec, cond_vec)

                metric_tests[m_key] = {
                    "metric_label": m_label,
                    "reference_mean": (
                        round(sum(ref_vec) / len(ref_vec), 4) if ref_vec else 0.0
                    ),
                    "condition_mean": (
                        round(sum(cond_vec) / len(cond_vec), 4) if cond_vec else 0.0
                    ),
                    "mean_difference": t_res["mean_diff"],
                    "paired_t_test": t_res,
                    "wilcoxon_signed_rank": w_res,
                    "effect_size": d_res,
                }

            paired_comparisons[cond_key] = {
                "condition_key": cond_key,
                "condition_name": cond_name,
                "reference_condition": "Reference",
                "num_paired_runs": len(shared_keys),
                "metrics": metric_tests,
            }

        return paired_comparisons

    @staticmethod
    def _compute_overall_telemetry(
        records: List[ExperimentRunRecord],
    ) -> Dict[str, Any]:
        """Aggregates global runtime, API latency, token counts, and estimated cost across all runs."""
        return {
            "total_runs_executed": len(records),
            "total_processing_time_sec": round(
                sum(r.processing_time_sec for r in records), 3
            ),
            "total_api_calls": sum(r.api_call_count for r in records),
            "overall_mean_api_latency_ms": round(
                sum(r.total_api_latency_ms for r in records)
                / max(1, sum(r.api_call_count for r in records)),
                2,
            ),
            "total_prompt_tokens": sum(r.prompt_tokens for r in records),
            "total_completion_tokens": sum(r.completion_tokens for r in records),
            "total_api_tokens": sum(r.total_api_tokens for r in records),
            "total_estimated_cost_usd": round(
                sum(r.estimated_cost_usd for r in records), 6
            ),
        }

    @classmethod
    def evaluate_matrix(cls, records: List[ExperimentRunRecord]) -> Dict[str, Any]:
        """
        Computes per-condition descriptive statistics, paired comparisons against Reference,
        communication predictor correlations, and overall telemetry across all experiment records.
        """
        by_condition, present_conditions = cls._group_records_by_condition(records)
        condition_summaries = {
            cond_key: cls._summarize_single_condition(cond_key, by_condition[cond_key])
            for cond_key in present_conditions
        }
        paired_comparisons = cls._compute_paired_comparisons(
            by_condition, present_conditions
        )
        comm_predictors = cls.analyze_communication_predictors(records)
        overall_telemetry = cls._compute_overall_telemetry(records)

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "overall_telemetry": overall_telemetry,
            "condition_summaries": condition_summaries,
            "paired_comparisons_vs_reference": paired_comparisons,
            "communication_predictors_analysis": comm_predictors,
        }

    @classmethod
    def export_results(
        cls,
        records: List[ExperimentRunRecord],
        evaluation_report: Dict[str, Any],
        output_dir: str = "runs",
    ) -> Dict[str, str]:
        """
        Exports consolidated experiment deliverables (individual runs CSV, condition summary CSV,
        paired comparisons CSV, JSON evaluation report, and Markdown report) to `output_dir`.
        """
        os.makedirs(output_dir, exist_ok=True)
        runs_csv_path = os.path.join(output_dir, "experiment_runs_90.csv")
        summary_csv_path = os.path.join(output_dir, "condition_summary.csv")
        paired_csv_path = os.path.join(output_dir, "paired_comparisons.csv")
        json_report_path = os.path.join(
            output_dir, "statistical_evaluation_report.json"
        )
        md_report_path = os.path.join(output_dir, "experiment_report.md")

        # Export individual runs CSV
        if records:
            flat_rows = []
            for r in records:
                d = r.to_dict()
                d["decoding_parameters"] = json.dumps(d["decoding_parameters"])
                d["workload_distribution"] = json.dumps(d["workload_distribution"])
                d["communication_function_counts"] = json.dumps(
                    d["communication_function_counts"]
                )
                d["communication_function_distribution"] = json.dumps(
                    d["communication_function_distribution"]
                )
                d["failure_categories"] = json.dumps(d["failure_categories"])
                flat_rows.append(d)

            fieldnames = list(flat_rows[0].keys())
            with open(runs_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)

        # Export per-condition summary CSV (Mean, SD, 95% CI)
        cond_summaries = evaluation_report.get("condition_summaries", {})
        summary_rows = []
        for cond_key, c_data in cond_summaries.items():
            for m_key, m_stats in c_data.get("metrics", {}).items():
                summary_rows.append(
                    {
                        "condition_key": cond_key,
                        "condition_name": c_data.get("condition_name", cond_key),
                        "metric": m_key,
                        "n": m_stats.get("n", 0),
                        "mean": m_stats.get("mean", 0.0),
                        "sd": m_stats.get("sd", 0.0),
                        "ci95_low": m_stats.get("ci95_low", 0.0),
                        "ci95_high": m_stats.get("ci95_high", 0.0),
                        "ci95_half_width": m_stats.get("ci95_half_width", 0.0),
                    }
                )
        if summary_rows:
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        # Export paired statistical comparisons CSV
        paired_comps = evaluation_report.get("paired_comparisons_vs_reference", {})
        paired_rows = []
        for cond_key, p_data in paired_comps.items():
            for m_key, m_test in p_data.get("metrics", {}).items():
                t_info = m_test.get("paired_t_test", {})
                w_info = m_test.get("wilcoxon_signed_rank", {})
                e_info = m_test.get("effect_size", {})
                paired_rows.append(
                    {
                        "condition_key": cond_key,
                        "condition_name": p_data.get("condition_name", cond_key),
                        "reference_condition": "Reference",
                        "metric": m_key,
                        "metric_label": m_test.get("metric_label", m_key),
                        "n_pairs": p_data.get("num_paired_runs", 0),
                        "reference_mean": m_test.get("reference_mean", 0.0),
                        "condition_mean": m_test.get("condition_mean", 0.0),
                        "mean_diff": m_test.get("mean_difference", 0.0),
                        "t_stat": t_info.get("t_stat", 0.0),
                        "t_p_value": t_info.get("p_value", 1.0),
                        "t_significant_p05": t_info.get("significant_p05", False),
                        "wilcoxon_w": w_info.get("w_stat", 0.0),
                        "wilcoxon_z": w_info.get("z_stat", 0.0),
                        "wilcoxon_p_value": w_info.get("p_value", 1.0),
                        "wilcoxon_significant_p05": w_info.get(
                            "significant_p05", False
                        ),
                        "cohens_d_pooled": e_info.get("cohens_d_pooled", 0.0),
                        "cohens_d_paired": e_info.get("cohens_d_paired", 0.0),
                        "effect_magnitude": e_info.get("magnitude", "negligible"),
                    }
                )
        if paired_rows:
            with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
                writer.writeheader()
                writer.writerows(paired_rows)

        # Export full JSON statistical evaluation report
        full_json = {
            **evaluation_report,
            "runs": [r.to_dict() for r in records],
        }
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(full_json, f, indent=2, ensure_ascii=False)

        # Export Markdown report with consolidated tables
        md_content = cls.format_markdown_report(evaluation_report)
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return {
            "runs_csv": runs_csv_path,
            "condition_summary_csv": summary_csv_path,
            "paired_comparisons_csv": paired_csv_path,
            "statistical_report_json": json_report_path,
            "markdown_report": md_report_path,
        }

    @classmethod
    def format_markdown_report(cls, evaluation_report: Dict[str, Any]) -> str:
        """Generates a comprehensive Markdown report with tables covering all experimental evaluation requirements."""
        cond_summaries = evaluation_report.get("condition_summaries", {})
        paired_comps = evaluation_report.get("paired_comparisons_vs_reference", {})
        comm_preds = evaluation_report.get("communication_predictors_analysis", {})
        telemetry = evaluation_report.get("overall_telemetry", {})

        lines = [
            "# Experimental Evaluation & Statistical Analysis Report",
            "",
            f"- **Total Executions:** {telemetry.get('total_runs_executed', 0)} runs (6 Tasks × 5 Constraint Conditions × 3 Repetitions)",
            f"- **Total Processing Time:** {telemetry.get('total_processing_time_sec', 0.0):.2f}s | **Mean API Latency:** {telemetry.get('overall_mean_api_latency_ms', 0.0):.1f} ms",
            f"- **Total API Tokens:** {telemetry.get('total_api_tokens', 0):,} (Prompt: {telemetry.get('total_prompt_tokens', 0):,}, Completion: {telemetry.get('total_completion_tokens', 0):,})",
            f"- **Estimated Total Cost:** \\${telemetry.get('total_estimated_cost_usd', 0.0):.4f} USD",
            "",
            "---",
            "",
            "## 1. Task-Performance & Coordination Metrics by Condition (Mean ± SD [95% CI])",
            "",
            "| Condition | Runs | Success Rate | DAG Progress (%) | Makespan (s) | Valid Action Rate | Violation Rate | Conflicts | Recovery Rate | Workload Gini |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for cond_key, c_data in cond_summaries.items():
            m = c_data.get("metrics", {})
            sr = m.get("task_success_rate", {})
            dp = m.get("dag_progress_pct", {})
            mk = m.get("makespan", {})
            va = m.get("valid_action_rate", {})
            cv = m.get("constraint_violation_rate", {})
            rc = m.get("resource_conflicts", {})
            rr = m.get("recovery_rate", {})
            wg = m.get("workload_gini", {})
            lines.append(
                f"| **{c_data.get('condition_name', cond_key)}** | {c_data.get('num_runs', 0)} | "
                f"{sr.get('mean', 0)*100:.1f}% ± {sr.get('sd', 0)*100:.1f}% [{sr.get('ci95_low', 0)*100:.1f}, {sr.get('ci95_high', 0)*100:.1f}] | "
                f"{dp.get('mean', 0):.1f}% ± {dp.get('sd', 0):.1f}% | "
                f"{mk.get('mean', 0):.1f} ± {mk.get('sd', 0):.1f} [{mk.get('ci95_low', 0):.1f}, {mk.get('ci95_high', 0):.1f}] | "
                f"{va.get('mean', 0)*100:.1f}% ± {va.get('sd', 0)*100:.1f}% | "
                f"{cv.get('mean', 0)*100:.1f}% | "
                f"{rc.get('mean', 0):.2f} ± {rc.get('sd', 0):.2f} | "
                f"{rr.get('mean', 0)*100:.1f}% | "
                f"{wg.get('mean', 0):.3f} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Natural-Language Communication & Independent LLM Judge Metrics (Mean ± SD [95% CI])",
                "",
                "| Condition | Messages | Comm Tokens | Grounded Rate | Redundant Rate | Judge Relevance | Judge Clarity | Judge Grounding | Judge Utility | Judge Overall (1-5) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for cond_key, c_data in cond_summaries.items():
            m = c_data.get("metrics", {})
            nm = m.get("number_of_messages", {})
            ct = m.get("total_communication_tokens", {})
            gr = m.get("grounded_message_rate", {})
            rd = m.get("redundant_message_rate", {})
            jr = m.get("judge_relevance", {})
            jc = m.get("judge_clarity", {})
            jg = m.get("judge_grounding", {})
            ju = m.get("judge_coordination_utility", {})
            jo = m.get("judge_overall_score", {})
            lines.append(
                f"| **{c_data.get('condition_name', cond_key)}** | "
                f"{nm.get('mean', 0):.1f} ± {nm.get('sd', 0):.1f} | "
                f"{ct.get('mean', 0):.1f} ± {ct.get('sd', 0):.1f} | "
                f"{gr.get('mean', 0)*100:.1f}% ± {gr.get('sd', 0)*100:.1f}% | "
                f"{rd.get('mean', 0)*100:.1f}% ± {rd.get('sd', 0)*100:.1f}% | "
                f"{jr.get('mean', 0):.2f} ± {jr.get('sd', 0):.2f} | "
                f"{jc.get('mean', 0):.2f} ± {jc.get('sd', 0):.2f} | "
                f"{jg.get('mean', 0):.2f} ± {jg.get('sd', 0):.2f} | "
                f"{ju.get('mean', 0):.2f} ± {ju.get('sd', 0):.2f} | "
                f"**{jo.get('mean', 0):.2f}** [{jo.get('ci95_low', 0):.2f}, {jo.get('ci95_high', 0):.2f}] |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Paired Statistical Comparisons vs. Reference Condition (Paired t-test, Wilcoxon Signed-Rank & Cohen's $d$)",
                "",
                "| Limitation Condition | Metric | Ref Mean | Cond Mean | Diff ($\\Delta$) | Paired $t$ ($p$-value) | Wilcoxon $W$ ($p$-value) | Cohen's $d$ (Magnitude) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        highlight_metrics = [
            "task_success_rate",
            "dag_progress_pct",
            "makespan",
            "valid_action_rate",
            "resource_conflicts",
            "number_of_messages",
            "total_communication_tokens",
            "grounded_message_rate",
            "redundant_message_rate",
            "judge_overall_score",
        ]
        for cond_key, p_data in paired_comps.items():
            c_name = p_data.get("condition_name", cond_key)
            m_map = p_data.get("metrics", {})
            for mk in highlight_metrics:
                if mk not in m_map:
                    continue
                mt = m_map[mk]
                t_info = mt.get("paired_t_test", {})
                w_info = mt.get("wilcoxon_signed_rank", {})
                e_info = mt.get("effect_size", {})
                sig_star = "*" if t_info.get("significant_p05") else ""
                w_star = "*" if w_info.get("significant_p05") else ""
                lines.append(
                    f"| **{c_name}** | {mt.get('metric_label', mk)} | "
                    f"{mt.get('reference_mean', 0):.3f} | {mt.get('condition_mean', 0):.3f} | "
                    f"{mt.get('mean_difference', 0):+.3f} | "
                    f"t={t_info.get('t_stat', 0):.2f} (p={t_info.get('p_value', 1):.4f}){sig_star} | "
                    f"W={w_info.get('w_stat', 0):.1f} (p={w_info.get('p_value', 1):.4f}){w_star} | "
                    f"d={e_info.get('cohens_d_pooled', 0):+.2f} ({e_info.get('magnitude', 'negligible')}) |"
                )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Failure Categories, Processing Time, API Latency & Cost Breakdown by Condition",
                "",
                "| Condition | Precondition / Dep Errors | Resource Conflicts | Skill Mismatches | Comm Budget Blocks | Injected Failures | Processing Time (s) | Mean API Latency (ms) | Total API Tokens | Est. Cost (USD) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for cond_key, c_data in cond_summaries.items():
            fc = c_data.get("failure_categories_total", {})
            tel = c_data.get("telemetry_totals", {})
            lines.append(
                f"| **{c_data.get('condition_name', cond_key)}** | "
                f"{fc.get('dependency_precondition_error', 0)} | "
                f"{fc.get('resource_lock_conflict', 0)} | "
                f"{fc.get('capability_skill_mismatch', 0)} | "
                f"{fc.get('communication_budget_exceeded', 0)} | "
                f"{fc.get('injected_environment_or_slip_failure', 0)} | "
                f"{tel.get('total_processing_time_sec', 0.0):.2f}s | "
                f"{tel.get('mean_api_latency_ms', 0.0):.1f} ms | "
                f"{tel.get('total_api_tokens', 0):,} | "
                f"\\${tel.get('total_estimated_cost_usd', 0.0):.4f} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. Do Communication Characteristics Predict Task Success or Recovery?",
                "",
                "| Communication Characteristic | $r$ (Task Success) | $p$-value | $r$ (DAG Progress) | $r$ (Recovery Rate) | Mean (Successful Runs) | Mean (Unsuccessful Runs) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for feat_key, pred in comm_preds.items():
            lines.append(
                f"| **{pred.get('label', feat_key)}** | "
                f"{pred.get('predicts_task_success_r', 0.0):+.3f} | "
                f"{pred.get('predicts_task_success_p', 1.0):.4f} | "
                f"{pred.get('predicts_dag_progress_r', 0.0):+.3f} | "
                f"{pred.get('predicts_recovery_rate_r', 0.0):+.3f} | "
                f"{pred.get('mean_in_successful_runs', 0.0):.3f} | "
                f"{pred.get('mean_in_unsuccessful_runs', 0.0):.3f} |"
            )

        lines.append("")
        return "\n".join(lines)


__all__ = [
    "ExperimentMatrixConfig",
    "ExperimentRunRecord",
    "BatchExperimentRunner",
    "StatisticalAnalysisEngine",
]
