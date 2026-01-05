#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
import uuid

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_science.modules import run_discovery_experiment
from data_science.data_loading_scripts.excel_cld_loader_v3 import load_cld_from_excel
from data_science.force_neo4j_fix import fix_neo4j_connection

try:
    from watchfiles import run as watch_run
    WATCH_AVAILABLE = True
except Exception:
    WATCH_AVAILABLE = False


def setup_logging(log_dir: str, verbose: bool, timestamp: str, experiment_id: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"experiment_{timestamp}_{experiment_id}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    class _ExpFilter(logging.Filter):
        def __init__(self, exp_id: str):
            super().__init__()
            self.exp_id = exp_id
        def filter(self, record: logging.LogRecord) -> bool:
            # attach exp_id to every record
            if not hasattr(record, 'exp_id'):
                record.exp_id = self.exp_id
            return True

    exp_filter = _ExpFilter(experiment_id)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if not verbose else logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s [exp:%(exp_id)s] %(levelname)s - %(message)s"))
    ch.addFilter(exp_filter)
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [exp:%(exp_id)s] %(name)s - %(levelname)s - %(message)s"))
    fh.addFilter(exp_filter)
    logger.addHandler(fh)

    return log_path


def run_once(args) -> None:
    # Apply local Neo4j connection fix (mirror notebook behavior)
    try:
        fix_neo4j_connection()
    except Exception:
        logging.warning("Neo4j fix not applied; proceeding with defaults")

    # Optional: quick validation load to echo counts
    try:
        _ = load_cld_from_excel(args.excel, plot_graph=False, export_json=False)
    except Exception:
        pass

    effective_prefix = f"{args.output_prefix}_{args._timestamp}_{args._experiment_id}"
    effective_description = f"{args.description} [exp_id={args._experiment_id} ts={args._timestamp}]"

    # Parse LLM configs (allow JSON or provider:model shorthand)
    def _parse_llm_config(val):
        if val is None:
            return None
        if isinstance(val, dict):
            return val
        s = str(val).strip()
        if not s:
            return None
        if s.startswith('{'):
            try:
                return json.loads(s)
            except Exception:
                logging.warning("Failed to parse JSON for LLM config: %s", s)
                return s
        if ':' in s:
            provider, model = s.split(':', 1)
            return {"provider": provider.strip(), "model": model.strip()}
        return s

    gen_cfg = _parse_llm_config(args.generator)
    corr_cfg = _parse_llm_config(args.corruptor)
    judge_cfg = _parse_llm_config(args.judge)

    # Normalize judge models: allow space- or comma-separated
    judge_models_norm = None
    if args.judge_models:
        flat = []
        for itm in args.judge_models:
            parts = [p.strip() for p in str(itm).split(',') if p.strip()]
            flat.extend(parts)
        judge_models_norm = flat or None

    # Resolve parallel flag: default True; --no-parallel overrides; --parallel enables
    use_parallel = True
    if getattr(args, "no_parallel", False):
        use_parallel = False
    elif getattr(args, "parallel", False):
        use_parallel = True

    info = run_discovery_experiment(
        retrieved_session_id=(args.retrieved_session_id if args.retrieved_session_id else 'False'),
        excel_path=args.excel,
        export_json=not args.no_export_json,
        plot_graph=args.plot_excel_graph,
        target_variable_key=args.target_key,
        temporal_scale_key=args.temporal_key,
        spatial_scale_key=args.spatial_key,
        context=args.context,
        yaml_path=args.yaml,
        dev_mode=args.dev,
        generator_config=gen_cfg,
        corruptor_config=corr_cfg,
        judge_config=judge_cfg,
        corruption_rate=args.corruption_rate,
        judge_edges=args.judge_edges,
        judge_models=judge_models_norm,
        num_judges=args.num_judges,
        generator_temperature=args.gen_temp,
        generator_top_p=args.gen_top_p,
        corruptor_temperature=args.corr_temp,
        corruptor_top_p=args.corr_top_p,
        judge_temperature=args.judge_temp,
        judge_top_p=args.judge_top_p,
        parallel=use_parallel,
        max_workers=args.max_workers,
        load_env=True,
        plot_session_graph=args.plot_session_graph,
        plot_validation_graph=args.plot_validation_graph,
        only_cited_edges=args.only_cited_edges,
        only_consistent_edges=args.only_consistent_edges,
        filter_validation_nodes_to_session_nodes=not args.no_filter_validation_nodes,
        experiment_description=effective_description,
        output_json_prefix=effective_prefix,
        compare_variable_overlap=args.compare_variable_overlap,
        result_excel_path=args.result_excel_path,
        overide_target_variable=args.override_target,
        citation_search_provider=args.citation_search_provider,
        # CI metrics
        embedding_enable=not args.no_ci,
        ci_fetch_reuse=not args.no_ci_reuse,
        ci_chunk_chars_override=args.ci_chunk_chars,
        ci_overlap_ratio=args.ci_overlap,
        # Node comparison controls
        node_comparison_enable=not args.no_node_comparison,
        node_comparison_mode=args.node_comparison_mode,
        node_comparison_cosine_percentile=args.node_comparison_cosine_percentile,
        node_comparison_cosine_model=args.node_comparison_cosine_model,
        node_comparison_cosine_dimensions=args.node_comparison_cosine_dimensions,
    )
    logging.info("Run completed: %s", json.dumps(info, indent=2))
    # Print a concise, human-readable end-of-run summary with key artifacts & usage stats
    try:
        if isinstance(info, dict):
            excel_out = info.get("result_excel_path") or args.result_excel_path
            session_id = info.get("session_id")
            stats = info.get("stats", {}) or {}
            lm_stats = info.get("lm_stats", {}) or {}

            print("\n================ Experiment Summary ================")
            print(f"Experiment ID: {args._experiment_id}")
            print(f"Timestamp:     {args._timestamp}")
            if session_id:
                print(f"Graph/Session: {session_id}")
            if excel_out:
                print(f"Results Excel: {excel_out}")
            if stats:
                print(f"Counts        : variables={stats.get('variable_count')}  relationships={stats.get('relationship_count')}  judged={stats.get('judged_count')}")

            # Print execution/CI config highlights
            print("-- Config Highlights --")
            print(f"Parallel={not getattr(args, 'no_parallel', False) or getattr(args, 'parallel', False)}  max_workers={args.max_workers}  judges={args.num_judges}  judge_models={args.judge_models}")
            print(f"CI: enabled={not args.no_ci} reuse={not args.no_ci_reuse} chunk_chars={args.ci_chunk_chars} overlap={args.ci_overlap}")

            # Print token usage metrics for generator/corruptor/judge/embeddings
            def _print_component_stats(name: str, s: dict):
                if not isinstance(s, dict):
                    print(f"{name}: <no stats>")
                    return
                status_counts = s.get('status_counts')
                token_totals = s.get('token_totals')
                calls = s.get('inference_call_count')
                time_total = s.get('inference_time_total')
                print(f"{name}: calls={calls} time_total={time_total}")
                if token_totals is not None:
                    try:
                        # Compact single-line representation
                        print(f"  tokens={json.dumps(token_totals, separators=(',',':'))}")
                    except Exception:
                        print(f"  tokens={token_totals}")
                if status_counts:
                    print(f"  status={status_counts}")

            print("-- Token Usage --")
            _print_component_stats("generator", lm_stats.get("generator", {}))
            _print_component_stats("corruptor", lm_stats.get("corruptor", {}))
            _print_component_stats("judge", lm_stats.get("judge", {}))
            _print_component_stats("embeddings", lm_stats.get("embeddings", {}))
            print("===================================================\n")
    except Exception:
        # Never fail the run on summary print issues
        pass


def main():
    p = argparse.ArgumentParser(description="Run CLD discovery without Jupyter, with rolling logs.")
    p.add_argument("--excel", required=True, help="Path to ground truth Excel CLD file")
    p.add_argument("--yaml", default="../backend/configs/prompts.yaml", help="Prompts YAML path")
    p.add_argument("--retrieved-session-id", dest="retrieved_session_id", default=None, help="Reuse existing Neo4j session id")
    p.add_argument("--output-prefix", dest="output_prefix", default="run", help="Prefix for output JSON filename")
    p.add_argument("--result-excel-path", dest="result_excel_path", default="results.xlsx", help="Path for results Excel")
    p.add_argument("--description", default="cli_run", help="Experiment description")
    p.add_argument("--context", default="", help="Additional context string passed into discovery")
    p.add_argument("--corruption-rate", dest="corruption_rate", type=float, default=0.0)

    # Context keys
    p.add_argument("--target-key", dest="target_key", default="Target")
    p.add_argument("--temporal-key", dest="temporal_key", default="Temporal Scale")
    p.add_argument("--spatial-key", dest="spatial_key", default="Spatial Scale")
    p.add_argument("--override-target", dest="override_target", default=None, help="Override target variable (empty string to disable target)")

    # LLM configs
    p.add_argument("--generator", default="PERPLEXITY_MODEL", help='Generator config: JSON or "provider:model" or model string')
    p.add_argument("--corruptor", default="PERPLEXITY_MODEL", help='Corruptor config: JSON or "provider:model" or model string')
    p.add_argument("--judge", default="CLAUDE_MODEL", help='Judge config: JSON or "provider:model" or model string')
    p.add_argument("--dev", action="store_true")
    p.add_argument("--gen-temp", dest="gen_temp", type=float, default=0.7)
    p.add_argument("--gen-top-p", dest="gen_top_p", type=float, default=1.0)
    p.add_argument("--corr-temp", dest="corr_temp", type=float, default=0.7)
    p.add_argument("--corr-top-p", dest="corr_top_p", type=float, default=1.0)
    p.add_argument("--judge-temp", dest="judge_temp", type=float, default=0.7)
    p.add_argument("--judge-top-p", dest="judge_top_p", type=float, default=1.0)
    p.add_argument("--judge-edges", dest="judge_edges", action="store_true")
    p.add_argument("--judge-models", nargs="*", default=None)
    p.add_argument("--num-judges", dest="num_judges", type=int, default=3)
    p.add_argument("--citation-search-provider", choices=["brave", "perplexity", None], default=None)

    # Execution
    p.add_argument("--parallel", action="store_true", help="Enable parallel discovery (default true)")
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel discovery")
    p.add_argument("--max-workers", type=int, default=3)
    p.add_argument("--plot-excel-graph", action="store_true")
    p.add_argument("--plot-session-graph", action="store_true")
    p.add_argument("--plot-validation-graph", action="store_true")
    p.add_argument("--no-export-json", dest="no_export_json", action="store_true", help="Disable exporting CLD JSONs when loading Excel")
    p.add_argument("--only-cited-edges", action="store_true")
    p.add_argument("--only-consistent-edges", action="store_true")
    p.add_argument("--no-filter-validation-nodes", action="store_true")
    p.add_argument("--compare-variable-overlap", action="store_true")

    # CI metrics controls
    p.add_argument("--no-ci", action="store_true", help="Disable context-insensitive metrics")
    p.add_argument("--no-ci-reuse", action="store_true", help="Do not reuse citation text from judge step")
    p.add_argument("--ci-chunk-chars", dest="ci_chunk_chars", type=int, default=None)
    p.add_argument("--ci-overlap", dest="ci_overlap", type=float, default=0.2)

    # Node comparison controls
    p.add_argument("--no-node-comparison", dest="no_node_comparison", action="store_true", help="Skip node and edge comparison step")
    p.add_argument("--node-comparison-mode", choices=["llm", "cosine", "hybrid"], default="llm")
    p.add_argument("--node-comparison-cosine-percentile", type=float, default=85.0)
    p.add_argument("--node-comparison-cosine-model", default="text-embedding-3-small")
    p.add_argument("--node-comparison-cosine-dimensions", type=int, default=None)

    # Watch & logging
    p.add_argument("--watch", action="store_true", help="Watch modules.py for changes and re-run")
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()

    # Load environment similarly to notebook behavior
    try:
        from dotenv import load_dotenv
        if os.path.exists('.env.dev'):
            load_dotenv('.env.dev', override=True)
        elif os.path.exists('.env copy'):
            load_dotenv('.env copy', override=True)
    except Exception:
        pass
    # Create static identifiers for this CLI session
    args._timestamp = time.strftime("%Y%m%d_%H%M%S")
    args._experiment_id = uuid.uuid4().hex[:8]
    log_file = setup_logging(args.log_dir, args.verbose, args._timestamp, args._experiment_id)
    logging.info("Logging to %s", log_file)
    logging.info("Experiment identifiers: exp_id=%s ts=%s", args._experiment_id, args._timestamp)

    if args.watch and not WATCH_AVAILABLE:
        logging.warning("--watch requested but watchfiles is not installed. Proceeding without watch.")
        args.watch = False

    def _runner():
        try:
            run_once(args)
        except Exception:
            logging.exception("Run failed")

    if not args.watch:
        _runner()
        return

    modules_path = os.path.join(ROOT, "data_science", "modules.py")
    logging.info("Watching %s for changes (Ctrl+C to stop)", modules_path)

    # First run
    _runner()

    if args.watch:
        # Re-run when modules.py changes
        def _on_change():
            logging.info("Change detected. Re-running...")
            _runner()

        # watchfiles runner blocks and calls target on change
        watch_run(_on_change, [modules_path])


if __name__ == "__main__":
    main()


