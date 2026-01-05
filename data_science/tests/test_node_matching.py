import os
import sys
import types
import json
import networkx as nx

import pytest
def _inject_fake_pydantic_ai(monkeypatch):
    fake_module = types.ModuleType("pydantic_ai")
    class FakeAgent:
        def __init__(self, *a, **k):
            pass
        def system_prompt(self, fn):
            return fn
        def run_sync(self, prompt, deps):
            raise RuntimeError("Agent.run_sync should be monkeypatched in tests")
    class FakeRunContext:
        pass
    fake_module.Agent = FakeAgent
    fake_module.RunContext = FakeRunContext
    sys.modules["pydantic_ai"] = fake_module


def _make_validation_json(tmp_path, nodes, edges):
    vars_path = tmp_path / "vars.json"
    edges_path = tmp_path / "edges.json"
    with open(vars_path, "w") as f:
        json.dump({"unique_variables": nodes}, f)
    with open(edges_path, "w") as f:
        json.dump({"edges": [{"source": s, "target": t} for s, t in edges]}, f)
    return str(vars_path), str(edges_path)


class DummyDiscovery:
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.session_id = "test-session"


def _monkey_extract_networkx_from_session(monkeypatch, G):
    import data_science.modules as modules
    def fake_extract(**kwargs):
        return G
    monkeypatch.setattr(modules, "extract_networkx_from_session", lambda **kw: fake_extract(**kw))


def _monkey_openai_embed_batch(monkeypatch, embedding_map):
    # embedding_map: dict name->vector(list)
    import data_science.logit_metrics as lm
    def fake_batch(texts, api_key, *, model="text-embedding-3-small", dimensions=None, timeout=30):
        vecs = []
        for t in texts:
            key = t.split("|")[0].strip()
            vecs.append(lm.np.array(embedding_map.get(key, embedding_map.get(key.lower(), [1.0, 0.0])), dtype=float))
        return vecs
    monkeypatch.setattr(lm, "openai_embed_batch", fake_batch)


def _monkey_llm_equiv(monkeypatch, predicate):
    # Patch Agent.run_sync to return object with data.MeansEquivalent
    import data_science.modules as modules
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
        def run_sync(self, prompt, deps):
            return Result(predicate(deps.target, deps.query))
    monkeypatch.setattr(modules, "Agent", FakeAgent)


@pytest.mark.parametrize("mode", ["llm", "cosine", "hybrid"])
def test_node_matching_basic(tmp_path, monkeypatch, mode):
    # Ensure embeddings path is active
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    # Pre-inject pydantic_ai shim before importing modules
    _inject_fake_pydantic_ai(monkeypatch)
    # Session graph: A->B
    Gs = nx.DiGraph()
    Gs.add_nodes_from(["A", "B"]) 
    Gs.add_edge("A", "B")
    discovery = DummyDiscovery(Gs)
    _monkey_extract_networkx_from_session(monkeypatch, Gs)

    # Validation graph: a->b (same semantics, different case)
    vs, ve = _make_validation_json(tmp_path, ["a", "b"], [("a", "b")])

    # Cosine embeddings: map A~a, B~b strongly; others weak
    embedding_map = {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
    }
    _monkey_openai_embed_batch(monkeypatch, embedding_map)

    # LLM predicate: equal ignoring case
    _monkey_llm_equiv(monkeypatch, lambda x, y: x.lower() == y.lower())

    from data_science.modules import compare_session_graph_to_validation

    m = compare_session_graph_to_validation(
        discovery,
        session_ids=discovery.session_id,
        validation_vars_json_path=vs,
        validation_edges_json_path=ve,
        plot_session_graph=False,
        plot_validation_graph=False,
        only_cited_edges=False,
        only_consistent_edges=False,
        filter_validation_nodes_to_session_nodes=False,
        node_match_mode=mode,
        cosine_percentile=50.0,
    )

    assert m["node_mapping"], "Expected a non-empty node mapping"
    # After mapping, edge should match
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    # Node F1 should be 1
    assert pytest.approx(m["node_f1"], rel=1e-6) == 1.0


def test_node_matching_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _inject_fake_pydantic_ai(monkeypatch)
    # Session: X->Y; Validation: A->B (distinct)
    Gs = nx.DiGraph()
    Gs.add_nodes_from(["X", "Y"]) 
    Gs.add_edge("X", "Y")
    discovery = DummyDiscovery(Gs)
    _monkey_extract_networkx_from_session(monkeypatch, Gs)

    vs, ve = _make_validation_json(tmp_path, ["A", "B"], [("A", "B")])

    # Embeddings make them orthogonal
    embedding_map = {
        "X": [1.0, 0.0],
        "Y": [0.0, 1.0],
        "A": [-1.0, 0.0],
        "B": [0.0, -1.0],
    }
    _monkey_openai_embed_batch(monkeypatch, embedding_map)
    # LLM predicate always false
    _monkey_llm_equiv(monkeypatch, lambda x, y: False)

    from data_science.modules import compare_session_graph_to_validation
    m = compare_session_graph_to_validation(
        discovery,
        session_ids=discovery.session_id,
        validation_vars_json_path=vs,
        validation_edges_json_path=ve,
        plot_session_graph=False,
        plot_validation_graph=False,
        filter_validation_nodes_to_session_nodes=False,
        node_match_mode="hybrid",
        cosine_percentile=90.0,
    )
    # No mapping => edges do not align
    assert len(m["node_mapping"]) == 0
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 1

