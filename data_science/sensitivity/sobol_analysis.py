import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
except Exception as e:  # pragma: no cover - handled in tests environment
    raise RuntimeError("SALib is required for sobol_analysis. Install SALib.") from e


@dataclass
class SobolConfig:
    problem: Dict
    base_sample_size: int
    calc_second_order: bool = False
    conf_level: float = 0.95
    n_resamples: int = 1000
    processes: Optional[int] = None
    seed: Optional[int] = None


class SobolRunner:
    """Run Saltelli sampling and Sobol analysis for a black-box evaluator.

    evaluator: function that maps a single 1D numpy array of length D to a scalar float.
    """

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], float],
        config: SobolConfig,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def sample(self) -> np.ndarray:
        # Use new SALib sobol.sample to support seeding and skip_values
        X = sobol_sample.sample(
            self.config.problem,
            N=self.config.base_sample_size,
            calc_second_order=self.config.calc_second_order,
            seed=self.config.seed,
        )
        return X

    def _eval_one(self, x: np.ndarray) -> float:
        return float(self.evaluator(x))

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.config.processes and self.config.processes > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(self.config.processes) as pool:
                Y_list = pool.map(self._eval_one, list(X))
        else:
            Y_list = [self._eval_one(x) for x in X]
        return np.asarray(Y_list, dtype=float)

    def analyze(self, Y: np.ndarray) -> Dict[str, np.ndarray]:
        # Use signature compatible args for installed SALib
        Si = sobol.analyze(
            self.config.problem,
            Y,
            calc_second_order=self.config.calc_second_order,
            num_resamples=self.config.n_resamples,
            conf_level=self.config.conf_level,
            print_to_console=False,
        )
        return Si

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        X = self.sample()
        Y = self.evaluate(X)
        Si = self.analyze(Y)
        return X, Y, Si


# --- Helper evaluators for tests/demo ---

def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> float:
    """Ishigami function with canonical parameters.
    Assumes x in [-pi, pi]^3.
    """
    x1, x2, x3 = x
    return math.sin(x1) + a * math.sin(x2) ** 2 + b * (x3 ** 4) * math.sin(x1)


def linear_first_only(x: np.ndarray) -> float:
    """Simple linear: depends only on first variable, others irrelevant."""
    return float(x[0])


# --- Adapter sketch for main_eval_single_run (not used in unit tests) ---

def build_problem_from_bounds(names: List[str], bounds: List[Tuple[float, float]]) -> Dict:
    assert len(names) == len(bounds), "names and bounds must be same length"
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }


def make_main_adapter(
    *,
    fixed_kwargs: Dict,
    metric_selector: Callable[[Dict], float],
    param_names: List[str],
) -> Callable[[np.ndarray], float]:
    """Return an evaluator that calls run_discovery_experiment with selected params.

    metric_selector(info_dict) -> float must extract the scalar metric.
    """
    from data_science.modules import run_discovery_experiment  # local import to avoid heavy import on module import

    def _evaluator(x: np.ndarray) -> float:
        overrides = {name: float(val) for name, val in zip(param_names, x)}
        kwargs = {**fixed_kwargs, **overrides}
        info = run_discovery_experiment(**kwargs)
        return float(metric_selector(info))

    return _evaluator
