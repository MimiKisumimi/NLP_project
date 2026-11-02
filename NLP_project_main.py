r"""
Multi-robot kitchen simulation CLI entrypoint (`NLP_project_main.py`).

Provides four execution modes via CLI flags:
1. Interactive / Single-Episode Live Simulation (default): runs a selected task,
   limitation condition, and failure scenario with Gemini agents and LLM Judge scoring.
2. Full 90-Run Batch Experiment Matrix (`--run-experiments`): runs the 6x5x3 factorial
   matrix and exports statistical evaluation reports.
3. Human-Judge Agreement Verification (`--verify-judge`): evaluates agreement and
   Pearson correlation between human ratings and the LLM Judge on sampled traces.
4. Interactive Gradio Trace Replay Dashboard (`--dashboard`): launches the web replay UI.
"""

import os
import sys
import time
import argparse
from typing import List

# Configure stdout and stderr to UTF-8 on Windows consoles so Unicode icons render cleanly.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import LimitationCondition, LimitationConfig, get_condition_config
from tasks import Task, TaskBenchmarkSuite
from metrics import (
    FailureType,
    FailureTrigger,
    FailureScenarioSuite,
    MetricsEngine,
)
from observer_judge import (
    LLMJudge,
    HumanJudgeAgreementVerifier,
)
from experiments import (
    ExperimentMatrixConfig,
    BatchExperimentRunner,
    StatisticalAnalysisEngine,
)
from dashboard import launch_dashboard
from environment import Environment
from agent import (
    GeminiProvider,
    RobotAgent,
    get_agent_profile,
    load_env,
)

# Load environment variables from .env if present
load_env()


def prompt_user_for_task(all_tasks: List[Task], default_index: int = 1) -> int:
    """Interactively prompts the user to select one benchmark task from the list of 6."""
    print("\n" + "=" * 70)
    print("Multi-Robot Kitchen Benchmark Suite - Select Task to Execute")
    print("=" * 70)
    for idx, tsk in enumerate(all_tasks, start=1):
        target_str = ", ".join(tsk.target_dishes)
        print(
            f"[{idx}] {tsk.name:34} | Difficulty: {tsk.difficulty:6} | Target: [{target_str}]"
        )
        print(f"    Description: {tsk.description}")
        if tsk.parallel_opportunities:
            print(f"    Coordination Focus: {tsk.parallel_opportunities[0]}")
        print()
    print("-" * 70)

    prompt_text = f"Select task to execute [1-{len(all_tasks)}] (Press Enter for default Task {default_index}): "
    while True:
        try:
            choice = input(prompt_text).strip()
            if not choice:
                print(
                    f">> Selected Task {default_index}: '{all_tasks[default_index - 1].name}'\n"
                )
                return default_index
            val = int(choice)
            if 1 <= val <= len(all_tasks):
                print(
                    f">> Selected Task {val}: '{all_tasks[val - 1].name}' ({all_tasks[val - 1].difficulty})\n"
                )
                return val
            print(
                f"⚠️ Invalid selection '{choice}'. Please enter a number between 1 and {len(all_tasks)}."
            )
        except (ValueError, EOFError):
            print(
                f">> Defaulting to Task {default_index}: '{all_tasks[default_index - 1].name}'\n"
            )
            return default_index


def prompt_user_for_condition(default_option: int = 2) -> LimitationConfig:
    """
    Interactively prompts the user to select one of the 5 controlled limitation conditions.
    """
    conditions = [
        (
            LimitationCondition.REFERENCE,
            "Reference",
            "Full visibility, generalist skills, unlimited communication",
        ),
        (
            LimitationCondition.CAPABILITY_LIMITED,
            "Capability-Limited",
            "Heterogeneous roles (Carrier, Manipulation, Heat-resistant)",
        ),
        (
            LimitationCondition.OBSERVATION_LIMITED,
            "Observation-Limited",
            "Partial visibility (local station only)",
        ),
        (
            LimitationCondition.COMMUNICATION_LIMITED,
            "Communication-Limited",
            "Capped message count & token budget",
        ),
        (
            LimitationCondition.COMBINED,
            "Combined",
            "All constraints combined (capability + visibility + communication budget)",
        ),
    ]
    print("\n" + "=" * 70)
    print("Robot Limitation Condition - Select Option [1-5]")
    print("=" * 70)
    for idx, (cond, name, desc) in enumerate(conditions, start=1):
        print(f"[{idx}] {name:24} | {desc}")
    print("-" * 70)

    prompt_text = (
        f"Select condition [1-5] "
        f"(Press Enter for default [{default_option}] {conditions[default_option - 1][1]}): "
    )
    while True:
        try:
            choice = input(prompt_text).strip()
            if not choice:
                selected_cond = conditions[default_option - 1][0]
                print(f">> Selected Condition: {conditions[default_option - 1][1]}\n")
                return get_condition_config(selected_cond)
            val = int(choice)
            if 1 <= val <= len(conditions):
                selected_cond = conditions[val - 1][0]
                print(f">> Selected Condition: {conditions[val - 1][1]}\n")
                return get_condition_config(selected_cond)
            print(
                f"⚠️ Invalid selection '{choice}'. Please enter a number between 1 and 5."
            )
        except (ValueError, EOFError):
            print(
                f">> Defaulting to Option {default_option}: '{conditions[default_option - 1][1]}'\n"
            )
            return get_condition_config(conditions[default_option - 1][0])


def prompt_user_for_failure(
    seed: int = 42, default_option: int = 1
) -> List[FailureTrigger]:
    """
    Interactively prompts the user to select if any controlled failure scenario(s)
    should be injected into the test before the simulation runs.
    """
    scenarios = FailureScenarioSuite.get_all_scenarios(seed=seed)
    print("\n" + "=" * 70)
    print("Failure & Recovery Scenario Injection - Select Option [1-7]")
    print("=" * 70)
    for idx, sc in enumerate(scenarios, start=1):
        timing = (
            "No failures injected"
            if sc.failure_type == FailureType.NONE
            else f"Triggers at t>={sc.trigger_timestep}s (Duration: {sc.duration}s)"
        )
        print(f"[{idx}] {sc.name:38} | {timing}")
        print(f"    Description: {sc.description}")
        print()
    print(
        f"[7] {'Combined Multi-Failure Stress Suite':38} | Triggers Tool + Slip + Msg Delay (Seed: {seed})"
    )
    print(
        f"    Description: Injects multiple predetermined failures across the episode (ideal for Medium/Hard tasks)."
    )
    print("-" * 70)

    prompt_text = (
        f"Select failure scenario(s) to inject [1-7, or comma-separated e.g. 2,4] "
        f"(Press Enter for default [{default_option}] No Failure): "
    )
    while True:
        try:
            choice = input(prompt_text).strip()
            if not choice:
                selected = FailureScenarioSuite.get_scenario_by_option(
                    default_option, seed=seed
                )
                label = (
                    scenarios[0].name
                    if default_option == 1
                    else f"Option {default_option}"
                )
                print(f">> Selected Failure Injection: {label}\n")
                return selected

            # Support comma-separated multi-select (e.g. "2,5") or single number 1-7
            parts = [p.strip() for p in choice.split(",") if p.strip()]
            if all(p.isdigit() and 1 <= int(p) <= 7 for p in parts):
                selected_triggers: List[FailureTrigger] = []
                selected_names: List[str] = []
                for p in parts:
                    opt = int(p)
                    if opt == 1:
                        selected_names.append(scenarios[0].name)
                    elif 2 <= opt <= 6:
                        trig = scenarios[opt - 1]
                        selected_triggers.append(trig)
                        selected_names.append(trig.name)
                    elif opt == 7:
                        combo = FailureScenarioSuite.get_scenario_by_option(
                            7, seed=seed
                        )
                        selected_triggers.extend(combo)
                        selected_names.append("Combined Multi-Failure Stress Suite")
                print(f">> Selected Failure Injection: {', '.join(selected_names)}\n")
                return selected_triggers

            print(
                "⚠️ Invalid selection. Please enter a number between 1 and 7 (or comma-separated numbers like '2,4')."
            )
        except (ValueError, EOFError):
            print(f">> Defaulting to Option 1: '{scenarios[0].name}'\n")
            return []


# ---------------------------------------------------------
# Main Simulation Execution (Gemini Default)
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Robot Kitchen Simulation Engine (Gemini Default)"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        choices=range(1, 7),
        help="Benchmark task index (1-6, default: 1)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="capability_limited",
        help="Controlled limitation condition (1-5 or name: reference, capability_limited, observation_limited, communication_limited, combined)",
    )
    parser.add_argument(
        "--failure",
        type=str,
        default=None,
        help="Controlled failure injection option (1-7 or name: none, tool_unavailable, workstation_blocked, action_execution_slip, misplaced_delivery, message_delay_or_drop, combined)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic failure scenarios (default: 42)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=45,
        help="Max simulation timesteps in seconds (default: 45)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Gemini model for robot agents (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Independent LLM Judge model (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--verify-judge",
        action="store_true",
        help="Run Human vs. LLM Judge agreement verification on 10-20 sampled message traces",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run Core 90-Run Experiment Execution Matrix (6 Tasks x 5 Conditions x 3 Repetitions) and Statistical Evaluation",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per (task, condition) cell in --run-experiments (default: 3 -> 90 runs)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not automatically launch the Gradio dashboard in the browser when simulation runs",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Gradio Trace Replay Dashboard directly",
    )
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard(runs_dir="runs")
        sys.exit(0)

    if args.run_experiments:
        print("===================================================================")
        print("Batch Experiment Harness & Statistical Evaluation")
        print("===================================================================")
        matrix_cfg = ExperimentMatrixConfig(
            repetitions=args.repetitions,
            llm_model=args.model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            judge_model=args.judge_model
            or os.getenv("JUDGE_LLM_MODEL", "gemini-3.1-pro-preview"),
            base_seed=args.seed,
            output_dir="runs",
        )
        print(
            f"Matrix Configuration: {len(matrix_cfg.task_indices)} Tasks x {len(matrix_cfg.conditions)} Conditions x {matrix_cfg.repetitions} Repetitions = {matrix_cfg.total_planned_runs()} Runs"
        )
        print(
            f"Robot Count:          {matrix_cfg.robot_count} Robots | Model: '{matrix_cfg.llm_model}' | Decoding: {matrix_cfg.decoding_parameters}"
        )
        print("-" * 67)
        runner = BatchExperimentRunner(
            config=matrix_cfg,
            verbose=False,
        )
        run_records = runner.run_all()
        eval_report = StatisticalAnalysisEngine.evaluate_matrix(run_records)
        exported_paths = StatisticalAnalysisEngine.export_results(
            records=run_records,
            evaluation_report=eval_report,
            output_dir=matrix_cfg.output_dir,
        )
        print("\n" + StatisticalAnalysisEngine.format_markdown_report(eval_report))
        print("===================================================================")
        print("Exported Consolidated Experiment Deliverables:")
        for k, p in exported_paths.items():
            print(f"  - {k:26}: {p}")
        print("===================================================================")
        sys.exit(0)

    if args.verify_judge:
        print("===================================================================")
        print("Human-Judge Agreement Verification Pipeline")
        print("===================================================================")
        agreement_data = HumanJudgeAgreementVerifier.run_verification_pipeline(
            runs_dir="runs",
            sample_size=12,
            worksheet_path="runs/human_judge_annotation_sample.json",
            report_output_path="runs/human_judge_agreement_report.json",
        )
        print(HumanJudgeAgreementVerifier.format_agreement_report(agreement_data))
        print(
            "\nSaved annotation worksheet to: runs/human_judge_annotation_sample.json"
        )
        print("Saved agreement report to:     runs/human_judge_agreement_report.json")
        sys.exit(0)

    print("===================================================================")
    print("Multi-Robot Kitchen Simulation Engine (Gemini Default)")
    print("===================================================================")

    # 1. Load Benchmark Suite
    all_tasks = TaskBenchmarkSuite.get_all_tasks()

    # Determine which task to run: prompt user interactively if --task not explicitly passed via CLI
    cli_specified_task = any(
        arg == "--task" or arg.startswith("--task=") for arg in sys.argv
    )
    if cli_specified_task:
        selected_task_idx = args.task
        print(
            f"\n[Task Selection] Specified via CLI argument: Task {selected_task_idx} ('{all_tasks[selected_task_idx - 1].name}')"
        )
    else:
        selected_task_idx = prompt_user_for_task(all_tasks, default_index=args.task)

    chosen_task = all_tasks[selected_task_idx - 1]

    # 1b. Determine Limitation Condition: prompt user interactively if --condition not passed via CLI
    cli_specified_condition = any(
        arg == "--condition" or arg.startswith("--condition=") for arg in sys.argv
    )
    if cli_specified_condition:
        active_cond_cfg = get_condition_config(args.condition)
        print(
            f"[Condition Selection] Specified via CLI argument: {active_cond_cfg.name} ({active_cond_cfg.condition.value})"
        )
    else:
        active_cond_cfg = prompt_user_for_condition(default_option=2)

    # 1c. Determine Failure Scenario Injection: prompt user interactively if --failure not passed via CLI
    cli_specified_failure = any(
        arg == "--failure" or arg.startswith("--failure=") for arg in sys.argv
    )
    if cli_specified_failure:
        val_str = str(args.failure).strip()
        active_failure_triggers = (
            FailureScenarioSuite.get_scenario_by_option(int(val_str), seed=args.seed)
            if val_str.isdigit()
            else FailureScenarioSuite.get_scenario_by_option(val_str, seed=args.seed)
        )
        trig_names = (
            [t.name for t in active_failure_triggers]
            if active_failure_triggers
            else ["No Failure Injection"]
        )
        print(f"[Failure Selection] Specified via CLI argument: {trig_names}")
    else:
        active_failure_triggers = prompt_user_for_failure(
            seed=args.seed, default_option=1
        )

    # 2. Check Gemini Provider Status & Model Selection
    gemini_check = GeminiProvider(model=args.model)
    has_api_key = gemini_check.is_available()

    print(f"[LLM Provider]  Google Gemini API (Default)")
    print(f"[Active Model] '{gemini_check.model}'")
    if has_api_key:
        print(
            f"[Gemini Status] 🟢 GEMINI_API_KEY detected (via environment or .env file)."
        )
    else:
        print(f"[Gemini Status] ⚠️ GEMINI_API_KEY is not set!")
        print("  To use the Google Gemini API for live agent reasoning:")
        print(
            "    1. Add your key to the '.env' file in this project: GEMINI_API_KEY=your_key"
        )
        print(
            "    2. Or set it in your terminal: $env:GEMINI_API_KEY='your_key' (PowerShell)"
        )
        print(
            "  (If GEMINI_API_KEY is missing, robots will report connection failure and idle each turn)"
        )

    # 3. Initialize Environment with Selected Task, Condition, and Failure Injection Triggers
    env = Environment(
        task=chosen_task,
        condition=active_cond_cfg,
        failure_triggers=active_failure_triggers,
        failure_seed=args.seed,
    )
    print(f"\n[Active Task]      {chosen_task.name} ({chosen_task.difficulty})")
    print(f"  Description:     {chosen_task.description}")
    print(
        f"  Target:          {chosen_task.target_dishes} at {chosen_task.target_station}"
    )
    print(
        f"[Active Condition] {active_cond_cfg.name} ({active_cond_cfg.condition.value})"
    )
    print(f"  Condition Rules: {active_cond_cfg.description}")
    if active_failure_triggers:
        print(
            f"[Failure Injection] {len(active_failure_triggers)} Controlled Failure(s) Armed (Seed={args.seed}):"
        )
        for ft in active_failure_triggers:
            print(f"  - {ft.name} ({ft.failure_type.value}): {ft.description}")
    else:
        print(f"[Failure Injection] None (Clean Run)")

    # 4. Initialize 3 Robots according to active LimitationCondition
    agents = []
    robot_start_stations = [
        ("Robot_1", "Refrigerator"),
        ("Robot_2", "Cutting table"),
        ("Robot_3", "Stove"),
    ]

    for r_id, start_pos in robot_start_stations:
        p = get_agent_profile(r_id, active_cond_cfg)
        robot = RobotAgent(
            r_id,
            skills=p["skills"],
            carrying_capacity=p["carrying_capacity"],
            tool_compatibility=p["tool_compatibility"],
            role=p["role"],
            description=p["description"],
            llm_provider=GeminiProvider(model=args.model),
            memory_limit=p["memory_limit"],
            max_messages=p["max_messages"],
            max_comm_tokens=p["max_comm_tokens"],
        )
        env.add_agent(robot, starting_location=start_pos)
        agents.append(robot)

    print(f"\n[Kitchen Setup] {len(env.items)} Items/Tools initialized:")
    for item in env.items.values():
        print(
            f"  - [{item.id}] {item.name:12} | Tool: {str(item.is_tool):5} | State: {item.state:10} | Location: {item.location}"
        )

    # 5. Start Gradio Replay Dashboard in browser as the simulation starts
    if not args.no_dashboard:
        if not env.observer:
            env.observer = PassiveObserver()
        if not env._observer_initialized:
            env.observer.observe_initialization(env)
            env._observer_initialized = True
        os.makedirs("runs", exist_ok=True)
        env.observer.export_trace("runs/live_run.json")

        print(f"\n===================================================================")
        print(f"🚀 Starting Gradio Dashboard in your browser as simulation starts...")
        print(f"👉 Web interface: http://localhost:7860")
        print(f"===================================================================")
        launch_dashboard(
            runs_dir="runs", port=7860, inbrowser=True, prevent_thread_lock=True
        )

    # Broadcast Task at t=0 and Run Live Simulation Loop with Gemini
    env.broadcast_task_at_start()
    max_steps = args.timesteps
    print(f"\n===================================================================")
    print(f"Starting Live Simulation Loop with Gemini (Max Timesteps: {max_steps}s)")
    print(f"Press Ctrl+C at any time to halt.")
    print(f"===================================================================")

    completed = False
    try:
        # Advance the simulation 1 second per tick until max_steps or task completion
        for t in range(1, max_steps + 1):
            env.timestep(t)
            if not args.no_dashboard:
                env.observer.export_trace("runs/live_run.json")
            if env.is_task_complete():
                completed = True
                print(
                    f"\n🎉 [SUCCESS] Ground-truth verifier confirmed: Task '{chosen_task.name}' completed at t={t}s!"
                )
                break
    except KeyboardInterrupt:
        print("\n[Simulation] Stopped by user (Ctrl+C).")

    # 6. Final Scorecard, Objective Metrics Report, Independent LLM Judge & Trace Export
    final_verif = env.verify_current_task()
    judge = LLMJudge(
        judge_model=args.judge_model,
        acting_agent_model=gemini_check.model,
    )
    task_metrics, comm_metrics, judge_report, saved_trace_path = (
        env.evaluate_and_export_trace(
            judge=judge,
        )
    )
    if not args.no_dashboard:
        env.observer.export_trace("runs/live_run.json")
    print("\n===================================================================")
    print("Simulation Run Summary")
    print("===================================================================")
    print(f"Run ID:                  {env.observer.run_id}")
    print(f"Trace Exported To:       {saved_trace_path}")
    print(f"Task:                    {chosen_task.name} ({chosen_task.difficulty})")
    print(
        f"Condition:               {active_cond_cfg.name} ({active_cond_cfg.condition.value})"
    )
    print(
        f"Target Dishes:           {chosen_task.target_dishes} at {chosen_task.target_station}"
    )
    print(f"Ground-Truth Complete:   {final_verif.is_complete}")
    print(f"Found Target Dishes:     {final_verif.target_dishes_found}")
    print(f"Missing Dishes:          {final_verif.missing_dishes}")
    print(MetricsEngine.format_metrics_report(task_metrics, comm_metrics))
    print(LLMJudge.format_judge_report(judge_report))

    # Keep dashboard server running so user can interactively inspect the run in their browser
    if not args.no_dashboard:
        print("\n" + "=" * 70)
        print(
            "🎉 Simulation finished! The Gradio Dashboard is active in your browser at:"
        )
        print("   👉 http://localhost:7860")
        print(
            "You can scrub through all timesteps, inspect the DAG, messages, and judge scores."
        )
        print("Click '🔄 Refresh Runs' in the dashboard to load the finalized trace.")
        print(
            "Press Ctrl+C in this terminal when you want to stop the dashboard server."
        )
        print("=" * 70)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Dashboard] Shutting down. Goodbye!")
