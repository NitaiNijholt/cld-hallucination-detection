import os
import sys
from typing import Any, Dict

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
# Keep other defaults local if not already set
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("ES_HOST", "localhost")
os.environ.setdefault("POSTGRES_HOST", "localhost")

from data_science.modules import run_discovery_experiment  # noqa: E402


def build_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def main() -> None:
    # Inputs/outputs relative to this script's directory
    excel_path: str = build_path("ground_truth_CLDs", "Social_norms_and_obesity_prevalence.xlsx")
    yaml_path: str = build_path("parameter_tuning_experiments", "alternative_prompts", "prompts_Nitai_C.yaml")
    result_excel_path: str = build_path(
        "gpt-4.1-brave-judge_gpt4.1-rerag-round_1-Social_norms_and_obesity_prevalence.xlsx"
    )

    # Model configs (fallbacks if not set via .env.dev)
    perplexity_model: str = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
    generator_model: str = os.environ.get("GENERATOR_MODEL", "gpt-4.1")
    judge_model: str = os.environ.get("JUDGE_MODEL", "gpt-4.1")

    # Create a unique per-run output directory
    import datetime, uuid
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    output_dir = build_path("runs", run_id)

    info: Dict[str, Any] = run_discovery_experiment(
        excel_path=excel_path,
        export_json=True,
        plot_graph=True,
        dev_mode=False,
        load_env=False,
        corruption_rate=0.0,
        judge_edges=True,
        judge_models=[judge_model],
        yaml_path=yaml_path,
        num_judges=1,
        generator_config={"provider": "openai", "model": generator_model},
        corruptor_config={"provider": "perplexity", "model": perplexity_model},
        judge_config={"provider": "openai", "model": judge_model},
        parallel=True,
        max_workers=3,
        plot_session_graph=True,
        plot_validation_graph=True,
        only_cited_edges=False,
        only_consistent_edges=False,
        filter_validation_nodes_to_session_nodes=True,
        experiment_description="judge_citation_per_test",
        output_json_prefix="judge_citation_per_test",
        compare_variable_overlap=False,
        result_excel_path=result_excel_path,
        overide_target_variable="",
        citation_search_provider="brave",
        output_dir=output_dir,
        # Disable node comparison to focus on CI cosine
        node_comparison_enable=False,
    )

    print("\nFinal Experiment Information:")
    print(info)
    print(f"Per-run output directory: {output_dir}")


if __name__ == "__main__":
    main()
