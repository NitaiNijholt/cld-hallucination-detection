import os
import sys
import json
import types
import tempfile
import networkx as nx


def inject_fake_pydantic_ai():
    fake_module = types.ModuleType("pydantic_ai")
    class FakeAgent:
        def __init__(self, *a, **k):
            pass
        def system_prompt(self, fn):
            return fn
        def run_sync(self, prompt, deps):
            raise RuntimeError("Agent.run_sync should be monkeypatched externally")
    class FakeRunContext:
        # Support subscription RunContext[T]
        def __class_getitem__(cls, item):
            return cls
    fake_module.Agent = FakeAgent
    fake_module.RunContext = FakeRunContext
    sys.modules["pydantic_ai"] = fake_module


def inject_backend_stubs():
    # Create minimal stub modules for heavy backend imports not needed for this test
    names = [
        "backend.citation_content_extractor",
        "backend.db_clients.neo4j_client",
        "backend.lm_clients.perplexity_client",
        "backend.prompt_tree",
        "backend.lm_clients.claude_client",
        "backend.lm_clients.openai_client_working",
        "data_loading_scripts.excel_cld_loader_v3",
        "memory_profiler",
    ]
    for name in names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
    # Provide a no-op profile decorator that accepts any kwargs
    def _noop_profile(*args, **kwargs):
        def _decorator(fn):
            return fn
        # support direct @profile without call syntax
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return _decorator
    sys.modules["memory_profiler"].profile = _noop_profile
    # Provide stub classes used by modules imports
    sys.modules["backend.db_clients.neo4j_client"].Neo4jClient = object
    sys.modules["backend.lm_clients.perplexity_client"].PerplexityClient = object
    sys.modules["backend.lm_clients.claude_client"].ClaudeClient = object
    sys.modules["backend.lm_clients.openai_client_working"].OpenAIClient = object
    sys.modules["backend.prompt_tree"].PromptTree = object
    sys.modules["backend.citation_content_extractor"].CitationProcessor = object
    sys.modules["data_loading_scripts.excel_cld_loader_v3"].load_cld_from_excel = lambda *a, **k: ([], [])


def write_validation(tmpdir, nodes, edges):
    vars_path = os.path.join(tmpdir, "vars.json")
    edges_path = os.path.join(tmpdir, "edges.json")
    with open(vars_path, "w") as f:
        json.dump({"unique_variables": nodes}, f)
    with open(edges_path, "w") as f:
        json.dump({"edges": [{"source": s, "target": t} for s, t in edges]}, f)
    return vars_path, edges_path


def main():
    os.environ.setdefault("OPENAI_API_KEY", "test")
    inject_fake_pydantic_ai()
    inject_backend_stubs()

    import data_science.modules as modules
    import data_science.logit_metrics as lm

    # Monkeypatch openai_embed_batch
    def fake_embed_batch(texts, api_key, *, model="text-embedding-3-small", dimensions=None, timeout=30):
        vecs = []
        for t in texts:
            key = t.split("|")[0].strip()
            if key.lower() in ("a", "b"):
                if key.lower() == "a":
                    vecs.append(lm.np.array([1.0, 0.0], dtype=float))
                else:
                    vecs.append(lm.np.array([0.0, 1.0], dtype=float))
            elif key in ("A", "B"):
                if key == "A":
                    vecs.append(lm.np.array([1.0, 0.0], dtype=float))
                else:
                    vecs.append(lm.np.array([0.0, 1.0], dtype=float))
            else:
                vecs.append(lm.np.array([1.0, 1.0], dtype=float))
        return vecs
    lm.openai_embed_batch = fake_embed_batch

    # Monkeypatch Agent to case-insensitive equality
    class Data:
        def __init__(self, eq):
            self.MeansEquivalent = eq
            self.Reasoning = "test"
    class Result:
        def __init__(self, eq):
            self.data = Data(eq)
    class FakeAgent:
        def __init__(self, *a, **k):
            pass
        def system_prompt(self, fn):
            return fn
        def run_sync(self, prompt, deps):
            return Result(deps.target.lower() == deps.query.lower())
    modules.Agent = FakeAgent

    # Build session graph
    Gs = nx.DiGraph()
    Gs.add_nodes_from(["A", "B"]) 
    Gs.add_edge("A", "B")

    class DummyDiscovery:
        def __init__(self, G):
            self.G = G
            self.session_id = "smoke"

    def fake_extract(**kwargs):
        return Gs
    modules.extract_networkx_from_session = lambda **kw: fake_extract(**kw)

    with tempfile.TemporaryDirectory() as tmpdir:
        vs, ve = write_validation(tmpdir, ["a", "b"], [("a", "b")])

        for mode in ("llm", "cosine", "hybrid"):
            metrics = modules.compare_session_graph_to_validation(
                DummyDiscovery(Gs),
                session_ids="smoke",
                validation_vars_json_path=vs,
                validation_edges_json_path=ve,
                plot_session_graph=False,
                plot_validation_graph=False,
                filter_validation_nodes_to_session_nodes=False,
                node_match_mode=mode,
                cosine_percentile=50.0,
            )
            print({
                "mode": mode,
                "node_mapping": metrics.get("node_mapping"),
                "node_f1": metrics.get("node_f1"),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
            })


if __name__ == "__main__":
    main()


