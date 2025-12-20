from __future__ import annotations

import argparse
import contextlib
import json
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

try:
    from ludic.agent import Agent
    from ludic.context import FullDialog
    from ludic.eval.core import run_eval_sync
    from ludic.inference import (
        InferenceSpec,
        ReturnSpec,
        SamplingParams,
        VLLMChatClient,
        start_vllm_server,
        wait_for_vllm_health,
    )
    from ludic.interaction import MultiAgentProtocol
    from ludic.training import EnvSpec, ProtocolSpec, Reducer, RolloutRequest
    from ludic.training.batching.rollout_engine import RolloutEngine
except ImportError as exc:
    raise SystemExit(
        "This script requires the optional 'ludic' dependency. "
        "Install with: uv sync --extra ludic"
    ) from exc

from sere.integrations.ludic_env import SereLudicEnv
from sere.integrations.ludic_parser import pddl_action_parser
from sere.io.task_loader import load_task


SERE_REDUCERS: Dict[str, Reducer] = {
    "success_rate": Reducer(
        kind="count_true",
        source=lambda rec: bool(rec.get("terminated")) and rec.get("outcome") == "success",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "invalid_rate": Reducer(
        kind="count_true",
        source=lambda rec: bool(rec.get("terminated")) and rec.get("outcome") == "invalid_move",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "truncation_rate": Reducer(
        kind="count_true",
        source=lambda rec: bool(rec.get("truncated")),
        normalize_by="rollouts",
        as_percent=True,
    ),
    "parse_error_rate": Reducer(
        kind="count_true",
        source=lambda rec: bool(rec.get("parse_error")),
        normalize_by="samples",
        as_percent=True,
    ),
    "avg_step_reward": Reducer(kind="mean", source="reward"),
    "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
}


@contextlib.contextmanager
def maybe_start_vllm(
    *,
    enabled: bool,
    model: str,
    host: str,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
) -> Iterable[None]:
    proc = None
    if enabled:
        proc = start_vllm_server(
            model,
            host,
            port,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=max_model_len,
        )
        wait_for_vllm_health(host, port, proc)
    try:
        yield None
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def build_inference(cfg: Dict[str, Any]) -> InferenceSpec:
    sampling = SamplingParams(
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens", 256)),
    )
    return InferenceSpec(sampling=sampling, return_=ReturnSpec.for_eval(return_token_ids=True))


def infer_agent_ids(
    *,
    task_path: str,
    domain_path: Optional[str],
    env_kwargs: Dict[str, Any],
) -> List[str]:
    env, _meta = load_task(domain_path, task_path, **env_kwargs)
    return SereLudicEnv(env).agent_ids


def build_requests(
    *,
    experiment: Dict[str, Any],
    global_env_defaults: Dict[str, Any],
    inference: InferenceSpec,
) -> List[RolloutRequest]:
    exp_name = experiment.get("name") or "unnamed"
    env_defaults = dict(global_env_defaults)
    env_defaults.update(experiment.get("env_defaults") or {})

    tasks = experiment.get("tasks") or []
    if not tasks:
        raise ValueError(f"Experiment '{exp_name}' has no tasks.")

    requests: List[RolloutRequest] = []
    for idx, task_cfg in enumerate(tasks):
        task_path = task_cfg.get("task") or task_cfg.get("path")
        if not task_path:
            raise ValueError(f"Experiment '{exp_name}' task[{idx}] is missing 'task'.")

        domain_path = task_cfg.get("domain_path") or task_cfg.get("domain")
        episodes = int(task_cfg.get("episodes", 1))

        env_kwargs = dict(env_defaults)
        env_kwargs.update(task_cfg.get("env") or {})
        env_kwargs.setdefault("run_mode", "interactive")

        agent_ids = infer_agent_ids(
            task_path=task_path,
            domain_path=domain_path,
            env_kwargs=env_kwargs,
        )

        env_spec_kwargs: Dict[str, Any] = {
            "task_path": task_path,
            "agent_ids": agent_ids,
            **env_kwargs,
        }
        if domain_path:
            env_spec_kwargs["domain_path"] = domain_path

        requests.append(
            RolloutRequest(
                env=EnvSpec(kind="sere", kwargs=env_spec_kwargs),
                protocol=ProtocolSpec(kind="multi_agent", kwargs={"agent_ids": agent_ids}),
                inference=inference,
                num_episodes=episodes,
                meta={
                    "experiment": exp_name,
                    "task": task_path,
                    "task_index": idx,
                },
            )
        )

    return requests


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Ludic agents on SERE tasks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/ludic/eval_config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment name(s) to run (repeatable).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output dir.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start a local vLLM server (overrides config).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    cfg = load_config(args.config)

    model = args.model if args.model is not None else cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
    host = args.host if args.host is not None else cfg.get("host", "127.0.0.1")
    port = int(args.port if args.port is not None else cfg.get("port", 8000))
    start_server = (
        bool(args.start_server)
        if args.start_server is not None
        else bool(cfg.get("start_server", False))
    )
    out_dir = args.out_dir if args.out_dir is not None else cfg.get("out_dir")

    inf_cfg = dict(cfg.get("inference") or {})
    if args.temperature is not None:
        inf_cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        inf_cfg["max_tokens"] = args.max_tokens
    inference = build_inference(inf_cfg)

    eval_cfg = dict(cfg.get("eval") or {})
    max_steps = int(args.max_steps if args.max_steps is not None else eval_cfg.get("max_steps", 40))
    timeout_s = args.timeout_s if args.timeout_s is not None else eval_cfg.get("timeout_s", None)
    concurrency = int(
        args.concurrency if args.concurrency is not None else eval_cfg.get("concurrency", 16)
    )

    vllm_gpu_util = float(
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else cfg.get("gpu_memory_utilization", 0.7)
    )
    max_model_len = (
        args.max_model_len if args.max_model_len is not None else cfg.get("max_model_len")
    )

    env_defaults = dict(cfg.get("env_defaults") or {})
    experiments = cfg.get("experiments") or []
    if not experiments:
        raise ValueError("Config must define at least one experiment.")

    if args.experiment:
        wanted = set(args.experiment)
        experiments = [exp for exp in experiments if exp.get("name") in wanted]
        if not experiments:
            raise ValueError(f"No experiments matched: {sorted(wanted)}")

    def env_factory(
        *,
        task_path: str,
        domain_path: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        **env_kwargs: Any,
    ) -> SereLudicEnv:
        env, _meta = load_task(domain_path, task_path, **env_kwargs)
        return SereLudicEnv(env, agent_ids=agent_ids)

    client = VLLMChatClient(host=host, port=port, enable_weight_updates=False)

    def protocol_factory(*, agent_ids: List[str]) -> MultiAgentProtocol:
        agents = {
            agent_id: Agent(
                client=client,
                model=model,
                ctx=FullDialog(),
                parser=pddl_action_parser(),
            )
            for agent_id in agent_ids
        }
        return MultiAgentProtocol(agents)

    engine = RolloutEngine(
        env_registry={"sere": env_factory},
        protocol_registry={"multi_agent": protocol_factory},
    )

    with maybe_start_vllm(
        enabled=start_server,
        model=model,
        host=host,
        port=port,
        gpu_memory_utilization=vllm_gpu_util,
        max_model_len=max_model_len,
    ):
        for exp in experiments:
            exp_name = exp.get("name") or "unnamed"
            requests = build_requests(
                experiment=exp,
                global_env_defaults=env_defaults,
                inference=inference,
            )

            records, metrics = run_eval_sync(
                engine=engine,
                requests=requests,
                reducers=SERE_REDUCERS,
                max_steps=max_steps,
                timeout_s=timeout_s,
                concurrency=concurrency,
            )

            print(f"\n---- SERE Evaluation: {exp_name} ----")
            for k, v in metrics.items():
                reducer = SERE_REDUCERS.get(k)
                if reducer and reducer.as_percent:
                    print(f"{k}={float(v):.2%}")
                else:
                    print(f"{k}={float(v):.4g}")

            if exp.get("out"):
                out_path = Path(exp["out"])
            elif out_dir:
                out_path = Path(out_dir) / f"{exp_name}.jsonl"
            else:
                out_path = None

            if out_path:
                write_jsonl(out_path, records)
                print(f"Wrote {len(records)} step records to {out_path}")


if __name__ == "__main__":
    main()
