import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import yaml

from data_science.sensitivity.sobol_analysis import (
    SobolConfig,
    SobolRunner,
    ishigami,
    build_problem_from_bounds,
)


def load_problem(path: str) -> Dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {
        "num_vars": len(data["names"]),
        "names": data["names"],
        "bounds": data["bounds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sobol sensitivity analysis")
    parser.add_argument("--problem", type=str, help="Path to YAML with names/bounds")
    parser.add_argument("--N", type=int, default=512, help="Base sample size")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--processes", type=int, default=0)
    parser.add_argument("--calc-second-order", action="store_true")
    parser.add_argument("--output", type=str, default="artifacts/sensitivity")
    parser.add_argument("--demo-ishigami", action="store_true", help="Run Ishigami demo instead of adapter")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo_ishigami:
        names = ["x1", "x2", "x3"]
        bounds = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
        problem = build_problem_from_bounds(names, bounds)
        evaluator = ishigami
    else:
        if not args.problem:
            raise SystemExit("--problem is required when not using --demo-ishigami")
        problem = load_problem(args.problem)
        raise SystemExit("Adapter mode requires supplying a custom evaluator; not included in CLI demo.")

    config = SobolConfig(
        problem=problem,
        base_sample_size=args.N,
        calc_second_order=args.calc_second_order,
        processes=(args.processes or None),
        seed=args.seed,
    )

    runner = SobolRunner(evaluator=evaluator, config=config)
    X, Y, Si = runner.run()

    out = {
        "S1": Si["S1"].tolist(),
        "ST": Si["ST"].tolist(),
        "S1_conf": Si.get("S1_conf", []).tolist() if "S1_conf" in Si else [],
        "ST_conf": Si.get("ST_conf", []).tolist() if "ST_conf" in Si else [],
    }

    out_path = os.path.join(args.output, "sobol_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
