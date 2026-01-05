import os
import sys
import argparse

# Ensure project root is on sys.path for imports like `data_science.modules`
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load environment
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.dev"), override=True)
except Exception:
    pass

# Force local Neo4j for development (override any docker hostnames from env/.env)
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_HOST"] = "localhost"
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("ES_HOST", "localhost")
os.environ.setdefault("POSTGRES_HOST", "localhost")

from data_science.modules import multirun_parameter_experiments  # noqa: E402


def build_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multirun parameter experiments")
    parser.add_argument(
        "--prompts-folder",
        dest="prompts_folder",
        default=build_path("parameter_tuning_experiments", "alternative_prompts"),
        help="Folder containing prompt YAMLs",
    )
    parser.add_argument(
        "--config",
        "--config-file",
        dest="config_file",
        default=build_path("parameter_tuning_experiments", "configs", "experiment_28_04_2025_config.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--cld-folder",
        dest="cld_folder",
        default=build_path("parameter_tuning_experiments", "ground_truth_clds_for_experiments"),
        help="Folder containing CLD Excel files",
    )
    parser.add_argument(
        "--results-folder",
        dest="results_folder",
        default=build_path("parameter_tuning_experiments", "results"),
        help="Folder to write results to",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if results already exist (skip existing result detection)",
    )
    args = parser.parse_args()

    def resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

    prompts_folder: str = resolve_path(args.prompts_folder)
    config_file: str = resolve_path(args.config_file)
    cld_folder: str = resolve_path(args.cld_folder)
    results_folder: str = resolve_path(args.results_folder)

    multirun_parameter_experiments(
        PROMPTS_FOLDER=prompts_folder,
        CONFIG_FILE=config_file,
        CLD_FOLDER=cld_folder,
        RESULTS_FOLDER=results_folder,
        FORCE_RERUN=args.force,
    )


if __name__ == "__main__":
    main()
