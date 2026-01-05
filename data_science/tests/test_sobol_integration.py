import types
import sys
import numpy as np
from data_science.sensitivity.sobol_analysis import (
    SobolConfig,
    SobolRunner,
    build_problem_from_bounds,
    make_main_adapter,
)


def test_adapter_with_stubbed_main():
    # Create a fake modules package with a stubbed run_discovery_experiment
    fake_modules = types.ModuleType("data_science.modules")

    # Define a deterministic scalar metric as a function of parameters: y = 2*x0 - x1
    def fake_run_discovery_experiment(**kwargs):
        x0 = float(kwargs.get("generator_temperature", 0.0))
        x1 = float(kwargs.get("generator_top_p", 0.0))
        return {"metric": 2.0 * x0 - 1.0 * x1}

    fake_modules.run_discovery_experiment = fake_run_discovery_experiment

    # Inject into sys.modules so make_main_adapter imports our fake instead of the heavy module
    sys.modules["data_science.modules"] = fake_modules

    names = ["generator_temperature", "generator_top_p"]
    bounds = [[0.0, 1.0], [0.0, 1.0]]
    problem = build_problem_from_bounds(names, bounds)

    evaluator = make_main_adapter(
        fixed_kwargs={},
        metric_selector=lambda info: info["metric"],
        param_names=names,
    )

    cfg = SobolConfig(problem=problem, base_sample_size=128, calc_second_order=False, seed=42)
    runner = SobolRunner(evaluator=evaluator, config=cfg)

    X, Y, Si = runner.run()

    # Expect x0 to contribute more than x1 due to coefficient 2 vs -1
    s1 = Si["S1"]
    st = Si["ST"]

    assert s1.shape[0] == 2
    assert s1[0] > s1[1]
    assert st[0] > st[1]
