from starlette.testclient import TestClient
import pytest
from server import app

def test_api():
    """Verify core API endpoints using in-process TestClient."""
    with TestClient(app) as client:
        # 1. Test /api/status
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "online"
        assert data["n_samples"] > 0
        assert data["stock_built"] is True
        assert data["adaptive_built"] is True

        # 2. Test /api/graph/projection
        r_graph = client.get("/api/graph/projection")
        assert r_graph.status_code == 200
        graph_data = r_graph.json()
        assert len(graph_data["nodes"]) > 0
        assert "edges_l0" in graph_data
        assert "edges_l1" in graph_data

        # 3. Test /api/search
        req_search = {"query_index": 0, "k": 5, "ef_search": 30}
        r_search = client.post("/api/search", json=req_search)
        assert r_search.status_code == 200
        search_data = r_search.json()
        assert "stock" in search_data and "adaptive" in search_data
        assert "recall" in search_data["stock"]
        assert "recall" in search_data["adaptive"]
        assert len(search_data["adaptive"]["trace"].get("steps", [])) > 0

        # 4. Test /api/semantic/search
        req_sem = {"query": "how does skip list hierarchy work?", "k": 2}
        r_sem = client.post("/api/semantic/search", json=req_sem)
        assert r_sem.status_code == 200
        sem_data = r_sem.json()
        assert len(sem_data["adaptive"]["hits"]) > 0
        assert "title" in sem_data["adaptive"]["hits"][0]
        assert "edge_savings_pct" in sem_data["comparison"]

if __name__ == '__main__':
    test_api()
    print("All endpoints verified successfully via TestClient!")
