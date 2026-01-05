import numpy as np
import math
from data_science.sensitivity.sobol_analysis import (
    SobolConfig,
    SobolRunner,
    ishigami,
    linear_first_only,
    build_problem_from_bounds,
)


def test_ishigami_small_N():
    # Canonical Ishigami setup in [-pi, pi]^3
    names = ["x1", "x2", "x3"]
    bounds = [[-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]]
    problem = build_problem_from_bounds(names, bounds)

    cfg = SobolConfig(problem=problem, base_sample_size=256, calc_second_order=False, seed=123)
    runner = SobolRunner(evaluator=lambda x: ishigami(x), config=cfg)
    X, Y, Si = runner.run()

    # Expected rough magnitudes: S1[x1] ~ 0.31, S1[x2] ~ 0.44, S1[x3] ~ 0
    s1 = Si["S1"]
    st = Si["ST"]
    assert s1.shape[0] == 3 and st.shape[0] == 3
    assert 0.15 < s1[0] < 0.5
    assert 0.2 < s1[1] < 0.7
    assert s1[2] < 0.2


def test_linear_first_only():
    # y = x1, others irrelevant in [0,1]
    names = ["x1", "x2", "x3"]
    bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    problem = build_problem_from_bounds(names, bounds)

    cfg = SobolConfig(problem=problem, base_sample_size=256, calc_second_order=False, seed=321)
    runner = SobolRunner(evaluator=linear_first_only, config=cfg)
    X, Y, Si = runner.run()

    s1 = Si["S1"]
    st = Si["ST"]
    # First variable dominates
    assert s1[0] > 0.8 and st[0] > 0.8
    # Others near zero
    assert s1[1] < 0.2 and st[1] < 0.2
    assert s1[2] < 0.2 and st[2] < 0.2
