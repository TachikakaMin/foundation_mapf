import glob
import math
import multiprocessing as mp
import os
import queue
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from tools.cached_distance_reader import read_distance_map_cached
from tools.extensions import (
    construct_input_feature_cpp as cpp_features,
    generate_lacam_solution_cpp,
)
from tools.utils import read_map


INT32_MAX = 2_147_483_647


@dataclass(frozen=True)
class _PairConfig:
    map_file: str
    map_name: str
    density: int
    agent_num: int
    weight: int


class MAPFOnlineDataset(IterableDataset):
    """
    Online MAPF training dataset.

    It samples (map, agent_num) pairs using the same agent-count weighting
    policy as gen_pathfile.sh, calls the C++ online solver bridge to generate
    one scenario, then yields all timestep samples from that scenario.
    """

    DEFAULT_AGENT_COUNTS = (256, 128, 96, 64, 32, 16)

    def __init__(
        self,
        map_files: Optional[Sequence[str]],
        feature_dim: int,
        feature_type: str,
        *,
        seed: int = 1919180,
        time_limit_sec: int = 2,
        verbose: int = 0,
        retry_limit: int = 64,
        agent_counts: Sequence[int] = DEFAULT_AGENT_COUNTS,
        buffer_size: int = 0,
        buffer_workers: int = 2,
        buffer_timeout_sec: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.feature_type = feature_type
        self.seed = int(seed)
        self.time_limit_sec = int(time_limit_sec)
        self.verbose = int(verbose)
        self.retry_limit = int(retry_limit)
        self.agent_counts = tuple(int(v) for v in agent_counts)
        self.buffer_size = int(buffer_size)
        self.buffer_workers = int(buffer_workers)
        self.buffer_timeout_sec = float(buffer_timeout_sec)

        if map_files is None:
            discovered = sorted(glob.glob("data/map_files/maze-*/*.map"))
            self.map_files = discovered
        else:
            self.map_files = sorted(str(p) for p in map_files)

        if not self.map_files:
            raise ValueError("No map files found for online dataset.")
        if self.retry_limit <= 0:
            raise ValueError("retry_limit must be > 0.")
        if not self.agent_counts:
            raise ValueError("agent_counts cannot be empty.")
        if self.buffer_size < 0:
            raise ValueError("buffer_size must be >= 0.")
        if self.buffer_workers < 0:
            raise ValueError("buffer_workers must be >= 0.")
        if self.buffer_size > 0 and self.buffer_workers <= 0:
            raise ValueError("buffer_workers must be > 0 when buffer_size > 0.")
        if self.buffer_size > 0 and self.buffer_timeout_sec <= 0:
            raise ValueError("buffer_timeout_sec must be > 0 when buffer_size > 0.")

        self._pairs = self._build_pair_configs(self.map_files, self.agent_counts)
        if not self._pairs:
            raise ValueError("No valid (map, agent_num) pair configurations.")

        weights = np.array([pair.weight for pair in self._pairs], dtype=np.float64)
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            raise ValueError("Pair weights sum to zero, cannot sample.")
        self._pair_probs = weights / weights_sum

        # Pair-specific base offsets keep seed streams distinct across pairs.
        self._pair_seed_bases = np.array(
            [(idx + 1) * 1_000_003 for idx in range(len(self._pairs))],
            dtype=np.int64,
        )

    def __iter__(self):
        worker_id, num_workers = self._get_worker_context()
        if self.buffer_size > 0 and self.buffer_workers > 0:
            yield from self._iter_buffered(worker_id, num_workers)
            return
        yield from self._iter_simple(worker_id, num_workers)

    def _iter_simple(self, worker_id: int, num_workers: int):
        worker_seed = self.seed + worker_id * 7_919
        rng = np.random.default_rng(worker_seed)

        # Each worker owns its own per-pair counter; the final seed additionally
        # interleaves workers by num_workers + worker_id so workers do not collide.
        pair_counters = np.zeros(len(self._pairs), dtype=np.int64)

        map_cache: Dict[str, np.ndarray] = {}
        distance_map_cache: Dict[str, object] = {}

        buffered_positions = None
        buffered_actions = None
        buffered_goals = None
        buffered_pair = None
        buffered_seed = None
        step_idx = 0
        prefetch_future = None

        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                if buffered_positions is not None and step_idx < buffered_positions.shape[0]:
                    yield self._build_step_sample(
                        pair=buffered_pair,
                        scenario_seed=int(buffered_seed),
                        positions=buffered_positions,
                        actions=buffered_actions,
                        goals=buffered_goals,
                        step_idx=step_idx,
                        map_cache=map_cache,
                        distance_map_cache=distance_map_cache,
                    )
                    step_idx += 1
                    continue

                buffered_positions = None
                buffered_actions = None
                buffered_goals = None
                buffered_pair = None
                buffered_seed = None
                step_idx = 0

                while buffered_positions is None:
                    if prefetch_future is None:
                        prefetch_future = self._submit_prefetch_task(
                            executor=executor,
                            rng=rng,
                            pair_counters=pair_counters,
                            worker_id=worker_id,
                            num_workers=num_workers,
                        )

                    prefetched = prefetch_future.result()
                    prefetch_future = None
                    if prefetched is None:
                        continue

                    (
                        buffered_pair,
                        buffered_seed,
                        buffered_positions,
                        buffered_actions,
                        buffered_goals,
                    ) = prefetched

                    prefetch_future = self._submit_prefetch_task(
                        executor=executor,
                        rng=rng,
                        pair_counters=pair_counters,
                        worker_id=worker_id,
                        num_workers=num_workers,
                    )

    def _iter_buffered(self, worker_id: int, num_workers: int):
        worker_seed = self.seed + worker_id * 7_919
        pair_counters = np.zeros(len(self._pairs), dtype=np.int64)
        pair_lock = threading.Lock()

        map_cache: Dict[str, np.ndarray] = {}
        distance_map_cache: Dict[str, object] = {}

        scenario_queue: "queue.Queue[object]" = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()
        error_holder: Dict[str, Optional[BaseException]] = {"exc": None}

        def producer_loop(producer_id: int):
            local_rng = np.random.default_rng(worker_seed + 104_729 * (producer_id + 1))
            while not stop_event.is_set():
                try:
                    attempts = self._prepare_attempts_threadsafe(
                        rng=local_rng,
                        pair_counters=pair_counters,
                        worker_id=worker_id,
                        num_workers=num_workers,
                        lock=pair_lock,
                    )
                    scenario = self._generate_scenario_from_attempts(attempts)
                    if scenario is None:
                        continue
                    while not stop_event.is_set():
                        try:
                            scenario_queue.put(scenario, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                except Exception as exc:
                    error_holder["exc"] = exc
                    stop_event.set()
                    try:
                        scenario_queue.put(exc, timeout=0.1)
                    except queue.Full:
                        pass
                    break

        threads: List[threading.Thread] = []
        for idx in range(self.buffer_workers):
            thread = threading.Thread(target=producer_loop, args=(idx,), daemon=True)
            thread.start()
            threads.append(thread)

        try:
            while True:
                try:
                    item = scenario_queue.get(timeout=self.buffer_timeout_sec)
                except queue.Empty:
                    if error_holder["exc"] is not None:
                        raise error_holder["exc"]
                    if stop_event.is_set():
                        raise RuntimeError("Online buffer stopped unexpectedly.")
                    continue
                if isinstance(item, Exception):
                    raise item
                pair, scenario_seed, positions, actions, goals = item
                for step_idx in range(positions.shape[0]):
                    yield self._build_step_sample(
                        pair=pair,
                        scenario_seed=int(scenario_seed),
                        positions=positions,
                        actions=actions,
                        goals=goals,
                        step_idx=step_idx,
                        map_cache=map_cache,
                        distance_map_cache=distance_map_cache,
                    )
        finally:
            stop_event.set()
            for thread in threads:
                thread.join(timeout=0.2)

    @staticmethod
    def _get_worker_context() -> Tuple[int, int]:
        worker = get_worker_info()
        if worker is None:
            return 0, 1
        return int(worker.id), int(worker.num_workers)

    @classmethod
    def _build_pair_configs(
        cls, map_files: Sequence[str], agent_counts: Sequence[int]
    ) -> List[_PairConfig]:
        pairs: List[_PairConfig] = []
        for map_file in map_files:
            map_name = os.path.splitext(os.path.basename(map_file))[0]
            density = cls._extract_density_from_map_name(map_name)
            for agent_num in agent_counts:
                weight = cls._offline_num_paths_weight(density, agent_num)
                if weight > 0:
                    pairs.append(
                        _PairConfig(
                            map_file=map_file,
                            map_name=map_name,
                            density=density,
                            agent_num=int(agent_num),
                            weight=int(weight),
                        )
                    )
        return pairs

    @staticmethod
    def _extract_density_from_map_name(map_name: str) -> int:
        # Expected: maze-32-32-10-1-75-0  -> density field is 10.
        parts = map_name.split("-")
        if len(parts) < 4:
            return 0
        try:
            return int(parts[3])
        except ValueError:
            return 0

    @staticmethod
    def _offline_num_paths_weight(density: int, agent_num: int) -> int:
        d = float(density)
        if agent_num == 256:
            value = 20 + d * 0.5
        elif agent_num == 128:
            value = 60 + d * 2.0
        elif agent_num == 96:
            value = 40 + d * 1.0
        elif agent_num == 64:
            value = 20 + d * 0.8
        elif agent_num == 32:
            value = 5 + d * 0.1
        elif agent_num == 16:
            value = 2 + d * 0.1
        else:
            return 0
        return int(math.floor(value + 0.5))

    @staticmethod
    def _normalize_seed(raw_seed: int) -> int:
        seed = int(raw_seed) % INT32_MAX
        if seed == 0:
            seed = 1
        return seed

    @classmethod
    def _make_worker_sharded_seed(
        cls,
        *,
        base_seed: int,
        pair_seed_base: int,
        pair_counter: int,
        worker_id: int,
        num_workers: int,
    ) -> int:
        # Interleave counters by worker id so workers never request same seed for
        # a given pair stream.
        raw = (
            int(base_seed)
            + int(pair_seed_base)
            + int(pair_counter) * int(num_workers)
            + int(worker_id)
            + 1
        )
        return cls._normalize_seed(raw)

    def _call_generator(self, map_file: str, agent_num: int, scenario_seed: int):
        # Prefer keyword call for readability; fall back to positional signatures.
        call_variants = (
            lambda: generate_lacam_solution_cpp(
                map_file=map_file,
                agent_num=int(agent_num),
                seed=int(scenario_seed),
                time_limit_sec=int(self.time_limit_sec),
                verbose=int(self.verbose),
            ),
            lambda: generate_lacam_solution_cpp(
                map_file,
                int(agent_num),
                int(scenario_seed),
                int(self.time_limit_sec),
                int(self.verbose),
            ),
            lambda: generate_lacam_solution_cpp(
                map_file, int(agent_num), int(scenario_seed), int(self.time_limit_sec)
            ),
            lambda: generate_lacam_solution_cpp(
                map_file, int(agent_num), int(scenario_seed)
            ),
        )

        last_type_error = None
        for call in call_variants:
            try:
                return call()
            except TypeError as exc:
                last_type_error = exc
                continue
            except RuntimeError:
                return None

        if last_type_error is not None:
            raise last_type_error
        return None

    def _prepare_attempts(
        self,
        *,
        rng: np.random.Generator,
        pair_counters: np.ndarray,
        worker_id: int,
        num_workers: int,
    ) -> List[Tuple[_PairConfig, int]]:
        attempts: List[Tuple[_PairConfig, int]] = []
        for _ in range(self.retry_limit):
            pair_idx = int(rng.choice(len(self._pairs), p=self._pair_probs))
            pair = self._pairs[pair_idx]

            local_counter = int(pair_counters[pair_idx])
            pair_counters[pair_idx] += 1

            scenario_seed = self._make_worker_sharded_seed(
                base_seed=self.seed,
                pair_seed_base=int(self._pair_seed_bases[pair_idx]),
                pair_counter=local_counter,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            attempts.append((pair, scenario_seed))
        return attempts

    def _prepare_attempts_threadsafe(
        self,
        *,
        rng: np.random.Generator,
        pair_counters: np.ndarray,
        worker_id: int,
        num_workers: int,
        lock: threading.Lock,
    ) -> List[Tuple[_PairConfig, int]]:
        attempts: List[Tuple[_PairConfig, int]] = []
        for _ in range(self.retry_limit):
            with lock:
                pair_idx = int(rng.choice(len(self._pairs), p=self._pair_probs))
                pair = self._pairs[pair_idx]
                local_counter = int(pair_counters[pair_idx])
                pair_counters[pair_idx] += 1

            scenario_seed = self._make_worker_sharded_seed(
                base_seed=self.seed,
                pair_seed_base=int(self._pair_seed_bases[pair_idx]),
                pair_counter=local_counter,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            attempts.append((pair, scenario_seed))
        return attempts

    def _generate_scenario_from_attempts(
        self,
        attempts: Sequence[Tuple[_PairConfig, int]],
    ):
        for pair, scenario_seed in attempts:
            raw = self._call_generator(
                map_file=pair.map_file,
                agent_num=pair.agent_num,
                scenario_seed=scenario_seed,
            )
            parsed = self._parse_solution_output(raw)
            if parsed is None:
                continue

            positions, actions, goals = parsed
            if positions.shape[0] == 0 or positions.shape[1] == 0:
                continue
            return pair, scenario_seed, positions, actions, goals
        return None

    def _submit_prefetch_task(
        self,
        *,
        executor: ThreadPoolExecutor,
        rng: np.random.Generator,
        pair_counters: np.ndarray,
        worker_id: int,
        num_workers: int,
    ):
        attempts = self._prepare_attempts(
            rng=rng,
            pair_counters=pair_counters,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        return executor.submit(self._generate_scenario_from_attempts, attempts)

    def _parse_solution_output(self, raw):
        if raw is None:
            return None

        if isinstance(raw, dict):
            success = raw.get("success", raw.get("ok", True))
            if not success:
                return None

            positions = None
            for key in (
                "positions",
                "agent_positions",
                "agent_locations",
                "paths",
                "trajectory",
                "solution",
            ):
                if key in raw:
                    positions = raw[key]
                    break

            actions = None
            for key in ("actions", "agent_actions", "action"):
                if key in raw:
                    actions = raw[key]
                    break

            goals = raw.get("goals", None)
            expected_steps = raw.get("steps", None)
            expected_agent_num = raw.get("agent_num", None)
            return self._normalize_solution_arrays(
                positions,
                actions,
                goals,
                expected_steps=expected_steps,
                expected_agent_num=expected_agent_num,
            )

        if isinstance(raw, (tuple, list)):
            if len(raw) == 0:
                return None

            if len(raw) >= 3 and isinstance(raw[0], (bool, np.bool_)):
                if not bool(raw[0]):
                    return None
                positions = raw[1]
                actions = raw[2]
                goals = raw[3] if len(raw) >= 4 else None
                return self._normalize_solution_arrays(
                    positions,
                    actions,
                    goals,
                    expected_steps=None,
                    expected_agent_num=None,
                )

            if len(raw) >= 2:
                positions = raw[0]
                actions = raw[1]
                goals = raw[2] if len(raw) >= 3 else None
                return self._normalize_solution_arrays(
                    positions,
                    actions,
                    goals,
                    expected_steps=None,
                    expected_agent_num=None,
                )

            return self._normalize_solution_arrays(
                raw[0],
                None,
                None,
                expected_steps=None,
                expected_agent_num=None,
            )

        return self._normalize_solution_arrays(
            raw,
            None,
            None,
            expected_steps=None,
            expected_agent_num=None,
        )

    def _normalize_solution_arrays(
        self,
        positions_raw,
        actions_raw,
        goals_raw,
        *,
        expected_steps,
        expected_agent_num,
    ):
        if positions_raw is None:
            return None

        positions = np.asarray(positions_raw, dtype=np.int64)
        if positions.ndim != 3 or positions.shape[2] != 2:
            return None

        steps, agent_num, _ = positions.shape
        if steps <= 0 or agent_num <= 0:
            return None
        if expected_steps is not None and int(expected_steps) != steps:
            return None
        if expected_agent_num is not None and int(expected_agent_num) != agent_num:
            return None

        if goals_raw is None:
            goals = positions[-1]
        else:
            goals = np.asarray(goals_raw, dtype=np.int64)
            if goals.ndim != 2 or goals.shape != (agent_num, 2):
                return None

        if actions_raw is None:
            actions = self._derive_actions_from_positions(positions)
            return positions, actions, goals

        actions = np.asarray(actions_raw, dtype=np.int64)
        if actions.ndim == 3 and actions.shape[2] == 1:
            actions = actions[:, :, 0]

        if actions.ndim != 2:
            return None
        if actions.shape[1] != agent_num:
            return None

        if actions.shape[0] == steps - 1:
            padded = np.zeros((steps, agent_num), dtype=np.int64)
            padded[:-1] = actions
            actions = padded
        elif actions.shape[0] != steps:
            return None

        return positions, actions, goals

    @staticmethod
    def _derive_actions_from_positions(positions: np.ndarray) -> np.ndarray:
        steps, agent_num, _ = positions.shape
        actions = np.zeros((steps, agent_num), dtype=np.int64)
        if steps <= 1:
            return actions

        delta = positions[1:] - positions[:-1]
        dx = delta[:, :, 0]
        dy = delta[:, :, 1]

        step_actions = np.zeros_like(dx, dtype=np.int64)
        step_actions[(dx == 0) & (dy == 1)] = 1
        step_actions[(dx == 0) & (dy == -1)] = 2
        step_actions[(dx == -1) & (dy == 0)] = 3
        step_actions[(dx == 1) & (dy == 0)] = 4
        actions[:-1] = step_actions
        return actions

    def _build_step_sample(
        self,
        *,
        pair: _PairConfig,
        scenario_seed: int,
        positions: np.ndarray,
        actions: np.ndarray,
        goals: np.ndarray,
        step_idx: int,
        map_cache: Dict[str, np.ndarray],
        distance_map_cache: Dict[str, object],
    ):
        map_data = map_cache.get(pair.map_file)
        if map_data is None:
            map_data = read_map(pair.map_file).astype(np.float32, copy=False)
            map_cache[pair.map_file] = map_data

        distance_map = distance_map_cache.get(pair.map_file)
        if distance_map is None:
            distance_map = read_distance_map_cached(pair.map_file)
            distance_map_cache[pair.map_file] = distance_map

        agent_locations_np = positions[step_idx].astype(np.int64, copy=False)
        goal_locations_np = goals.astype(np.int64, copy=False)

        input_features_np = cpp_features(
            map_data,
            agent_locations_np,
            goal_locations_np,
            distance_map,
            self.feature_dim,
            self.feature_type,
        )

        actions_np = actions[step_idx].astype(np.int64, copy=False)

        h, w = map_data.shape
        action_map = np.zeros((h, w), dtype=np.int64)
        action_map[agent_locations_np[:, 0], agent_locations_np[:, 1]] = actions_np

        mask_map = np.zeros((h, w), dtype=np.uint8)
        mask_map[agent_locations_np[:, 0], agent_locations_np[:, 1]] = 1

        file_name = (
            f"online/{pair.map_name}/{pair.map_name}-{pair.agent_num}-{scenario_seed}"
        )

        return {
            "feature": input_features_np,
            "action": action_map,
            "mask": mask_map,
            "file_name": file_name,
        }


def _scenario_producer_process(
    worker_idx: int,
    pairs: Sequence[_PairConfig],
    pair_probs: np.ndarray,
    pair_seed_bases: np.ndarray,
    base_seed: int,
    time_limit_sec: int,
    verbose: int,
    retry_limit: int,
    scenario_queue: mp.Queue,
    stop_event,
    seed_counter,
    seed_counter_lock,
):
    """独立进程：只做 LACAM 求解，把 raw scenario 数据发回主进程。
    Standalone process: runs LACAM only, sends raw scenario data back to main process."""
    from tools.extensions import generate_lacam_solution_cpp as _gen_lacam

    rng = np.random.default_rng(base_seed + (worker_idx + 1) * 104_729)
    map_cache: Dict[str, np.ndarray] = {}
    distance_map_cache: Dict[str, object] = {}

    def _normalize_seed(raw: int) -> int:
        s = int(raw) % INT32_MAX
        return s if s != 0 else 1

    def _next_seed() -> int:
        with seed_counter_lock:
            counter = seed_counter.value
            seed_counter.value += 1
        return _normalize_seed(base_seed + (worker_idx + 1) * 10_000_019 + counter)

    def _call_generator(map_file: str, agent_num: int, scenario_seed: int):
        call_variants = (
            lambda: _gen_lacam(
                map_file=map_file, agent_num=int(agent_num),
                seed=int(scenario_seed), time_limit_sec=int(time_limit_sec),
                verbose=int(verbose),
            ),
            lambda: _gen_lacam(
                map_file, int(agent_num), int(scenario_seed),
                int(time_limit_sec), int(verbose),
            ),
            lambda: _gen_lacam(
                map_file, int(agent_num), int(scenario_seed), int(time_limit_sec),
            ),
            lambda: _gen_lacam(map_file, int(agent_num), int(scenario_seed)),
        )
        last_type_error = None
        for call in call_variants:
            try:
                return call()
            except TypeError as exc:
                last_type_error = exc
                continue
            except RuntimeError:
                return None
        if last_type_error is not None:
            raise last_type_error
        return None

    def _parse_raw(raw):
        """轻量解析，只提取 positions/actions/goals 的 numpy 数组。"""
        if raw is None:
            return None
        if isinstance(raw, dict):
            success = raw.get("success", raw.get("ok", True))
            if not success:
                return None
            positions = None
            for key in ("positions", "agent_positions", "agent_locations",
                        "paths", "trajectory", "solution"):
                if key in raw:
                    positions = raw[key]
                    break
            actions = None
            for key in ("actions", "agent_actions", "action"):
                if key in raw:
                    actions = raw[key]
                    break
            goals = raw.get("goals", None)
            return positions, actions, goals
        if isinstance(raw, (tuple, list)):
            if len(raw) == 0:
                return None
            if len(raw) >= 3 and isinstance(raw[0], (bool, np.bool_)):
                if not bool(raw[0]):
                    return None
                return raw[1], raw[2], raw[3] if len(raw) >= 4 else None
            if len(raw) >= 2:
                return raw[0], raw[1], raw[2] if len(raw) >= 3 else None
            return raw[0], None, None
        return raw, None, None

    try:
        while not stop_event.is_set():
            for _ in range(retry_limit):
                pair_idx = int(rng.choice(len(pairs), p=pair_probs))
                pair = pairs[pair_idx]
                scenario_seed = _next_seed()
                raw = _call_generator(pair.map_file, pair.agent_num, scenario_seed)
                parsed = _parse_raw(raw)
                if parsed is None:
                    continue
                pos_raw, act_raw, goal_raw = parsed
                if pos_raw is None:
                    continue
                # 发送轻量数据：pair index + seed + raw numpy 数组
                try:
                    scenario_queue.put(
                        (pair_idx, scenario_seed, pos_raw, act_raw, goal_raw),
                        timeout=0.2,
                    )
                except Exception:
                    if stop_event.is_set():
                        return
                    continue
                break
    except BaseException:
        pass


class MAPFOnlineBufferLoader:
    """
    Global step-buffer loader for online MAPF training.

    使用多进程做 LACAM 求解（重活，释放 GIL 无效因为 Python 层开销），
    主进程单线程做 feature construction（轻活，~6ms/scenario）。
    Uses multiprocessing for LACAM solving (heavy work),
    main process does feature construction (lightweight, ~6ms/scenario).
    """

    def __init__(
        self,
        map_files: Optional[Sequence[str]],
        feature_dim: int,
        feature_type: str,
        *,
        batch_size: int,
        seed: int = 1919180,
        time_limit_sec: int = 2,
        verbose: int = 0,
        retry_limit: int = 64,
        agent_counts: Sequence[int] = MAPFOnlineDataset.DEFAULT_AGENT_COUNTS,
        buffer_size: int = 1024,
        buffer_workers: int = 4,
        buffer_timeout_sec: float = 1.0,
    ):
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.buffer_workers = int(buffer_workers)
        self.buffer_timeout_sec = float(buffer_timeout_sec)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must be >= batch_size for step buffering.")
        if self.buffer_workers <= 0:
            raise ValueError("buffer_workers must be > 0.")
        if self.buffer_timeout_sec <= 0:
            raise ValueError("buffer_timeout_sec must be > 0.")

        self._dataset = MAPFOnlineDataset(
            map_files,
            feature_dim,
            feature_type,
            seed=seed,
            time_limit_sec=time_limit_sec,
            verbose=verbose,
            retry_limit=retry_limit,
            agent_counts=agent_counts,
            buffer_size=0,
            buffer_workers=0,
            buffer_timeout_sec=buffer_timeout_sec,
        )

        # 进程间共享的 scenario queue（LACAM 结果）和 step queue（最终样本）
        self._scenario_queue: mp.Queue = mp.Queue(maxsize=self.buffer_workers * 4)
        self._step_queue: "queue.Queue[dict]" = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = mp.Event()
        self._seed_counter = mp.Value("i", 0)
        self._seed_counter_lock = mp.Lock()
        self._processes: List[mp.Process] = []
        self._num_builder_threads = 4
        self._builder_threads: List[threading.Thread] = []
        self._started = False
        self._start_lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        atexit.register(self.close)

    @property
    def pair_configs(self) -> Sequence[_PairConfig]:
        return self._dataset._pairs

    def start(self):
        with self._start_lock:
            if self._started:
                return
            self._started = True

            pairs = list(self._dataset._pairs)
            pair_probs = self._dataset._pair_probs.copy()
            pair_seed_bases = self._dataset._pair_seed_bases.copy()

            for worker_idx in range(self.buffer_workers):
                p = mp.Process(
                    target=_scenario_producer_process,
                    args=(
                        worker_idx,
                        pairs,
                        pair_probs,
                        pair_seed_bases,
                        self._dataset.seed,
                        self._dataset.time_limit_sec,
                        self._dataset.verbose,
                        self._dataset.retry_limit,
                        self._scenario_queue,
                        self._stop_event,
                        self._seed_counter,
                        self._seed_counter_lock,
                    ),
                    daemon=True,
                )
                p.start()
                self._processes.append(p)

            for _ in range(self._num_builder_threads):
                t = threading.Thread(target=self._builder_loop, daemon=True)
                t.start()
                self._builder_threads.append(t)

    def close(self):
        if not self._started:
            return
        self._started = False

        self._stop_event.set()
        self._scenario_queue.cancel_join_thread()

        for p in self._processes:
            if p.is_alive():
                p.kill()
        for p in self._processes:
            p.join(timeout=2.0)

        while not self._step_queue.empty():
            try:
                self._step_queue.get_nowait()
            except queue.Empty:
                break

        for t in self._builder_threads:
            t.join(timeout=2.0)

    def __iter__(self):
        self.start()
        batch_queue: "queue.Queue[dict]" = queue.Queue(maxsize=4)

        def _prefetch_loop():
            try:
                while not self._stop_event.is_set():
                    batch_samples = []
                    while len(batch_samples) < self.batch_size:
                        if self._stop_event.is_set():
                            return
                        sample = self._get_next_step_sample()
                        batch_samples.append(sample)
                    batch = self._collate_step_samples(batch_samples)
                    while not self._stop_event.is_set():
                        try:
                            batch_queue.put(batch, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception:
                pass

        prefetch_thread = threading.Thread(target=_prefetch_loop, daemon=True)
        prefetch_thread.start()

        while True:
            try:
                batch = batch_queue.get(timeout=self.buffer_timeout_sec)
                yield batch
            except queue.Empty:
                self._raise_if_error()
                if self._stop_event.is_set():
                    break
                continue

    def _set_error(self, exc: BaseException):
        with self._error_lock:
            if self._error is None:
                self._error = exc
        self._stop_event.set()

    def _raise_if_error(self):
        with self._error_lock:
            if self._error is not None:
                raise RuntimeError("Online step buffer producer failed.") from self._error

    def _get_next_step_sample(self):
        while not self._stop_event.is_set():
            self._raise_if_error()
            try:
                return self._step_queue.get(timeout=self.buffer_timeout_sec)
            except queue.Empty:
                continue
        self._raise_if_error()
        raise RuntimeError("Online step buffer stopped unexpectedly.")

    def _builder_loop(self):
        """主进程线程：从 scenario_queue 消费 raw scenario，做 normalize + feature construction，推入 step_queue。"""
        map_cache: Dict[str, np.ndarray] = {}
        distance_map_cache: Dict[str, object] = {}
        pairs = self._dataset._pairs

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._scenario_queue.get(timeout=self.buffer_timeout_sec)
                except Exception:
                    continue

                pair_idx, scenario_seed, pos_raw, act_raw, goal_raw = item
                pair = pairs[pair_idx]

                parsed = self._dataset._normalize_solution_arrays(
                    pos_raw, act_raw, goal_raw,
                    expected_steps=None, expected_agent_num=None,
                )
                if parsed is None:
                    continue
                positions, actions, goals = parsed
                if positions.shape[0] == 0 or positions.shape[1] == 0:
                    continue

                for step_idx in range(positions.shape[0]):
                    sample = self._dataset._build_step_sample(
                        pair=pair,
                        scenario_seed=int(scenario_seed),
                        positions=positions,
                        actions=actions,
                        goals=goals,
                        step_idx=step_idx,
                        map_cache=map_cache,
                        distance_map_cache=distance_map_cache,
                    )
                    while not self._stop_event.is_set():
                        try:
                            self._step_queue.put(sample, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:
            self._set_error(exc)

    @staticmethod
    def _collate_step_samples(samples: Sequence[dict]):
        feature = torch.from_numpy(np.stack([s["feature"] for s in samples], axis=0)).pin_memory()
        action = torch.from_numpy(np.stack([s["action"] for s in samples], axis=0)).pin_memory()
        mask = torch.from_numpy(np.stack([s["mask"] for s in samples], axis=0)).pin_memory()
        file_name = [s["file_name"] for s in samples]
        return {
            "feature": feature,
            "action": action,
            "mask": mask,
            "file_name": file_name,
        }
