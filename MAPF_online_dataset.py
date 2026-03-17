import glob
import math
import os
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

    DEFAULT_AGENT_COUNTS = (128, 96, 64, 32, 16)

    def __init__(
        self,
        map_files: Optional[Sequence[str]],
        feature_dim: int,
        feature_type: str,
        *,
        seed: int = 1919180,
        time_limit_sec: int = 5,
        verbose: int = 0,
        retry_limit: int = 64,
        agent_counts: Sequence[int] = DEFAULT_AGENT_COUNTS,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.feature_type = feature_type
        self.seed = int(seed)
        self.time_limit_sec = int(time_limit_sec)
        self.verbose = int(verbose)
        self.retry_limit = int(retry_limit)
        self.agent_counts = tuple(int(v) for v in agent_counts)

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

            # Need a new scenario.
            buffered_positions = None
            buffered_actions = None
            buffered_goals = None
            buffered_pair = None
            buffered_seed = None
            step_idx = 0

            got_one = False
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

                buffered_positions = positions
                buffered_actions = actions
                buffered_goals = goals
                buffered_pair = pair
                buffered_seed = scenario_seed
                got_one = True
                break

            if not got_one:
                raise RuntimeError(
                    f"Online dataset worker {worker_id} failed to generate a valid "
                    f"scenario after {self.retry_limit} retries."
                )

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
        if agent_num == 128:
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
        input_features = torch.from_numpy(input_features_np)

        agent_locations = torch.from_numpy(agent_locations_np)
        actions_t = torch.from_numpy(actions[step_idx].astype(np.int64, copy=False))

        output_features = torch.zeros(map_data.shape, dtype=torch.long)
        output_features[agent_locations[:, 0], agent_locations[:, 1]] = actions_t

        mask = torch.zeros(map_data.shape, dtype=torch.uint8)
        mask[agent_locations[:, 0], agent_locations[:, 1]] = 1

        file_name = (
            f"online/{pair.map_name}/{pair.map_name}-{pair.agent_num}-{scenario_seed}"
        )

        return {
            "feature": input_features,
            "action": output_features,
            "mask": mask,
            "file_name": file_name,
        }
