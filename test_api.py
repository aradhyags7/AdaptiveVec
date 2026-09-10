import urllib.request
import json

def test_api():
    print("Testing /api/status...")
    r = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/status').read().decode())
    print("Status:", r['status'], "Samples:", r['n_samples'])

    print("\nTesting /api/graph/projection...")
    r_graph = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/graph/projection').read().decode())
    print(f"Nodes: {len(r_graph['nodes'])}, L0 Edges: {len(r_graph['edges_l0'])}, L1 Edges: {len(r_graph['edges_l1'])}")

    print("\nTesting /api/search...")
    req_data = json.dumps({'query_index': 0, 'k': 5, 'ef_search': 30}).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:8000/api/search', data=req_data, headers={'Content-Type': 'application/json'})
    r_search = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"Stock Recall: {r_search['stock']['recall']}, Adaptive Recall: {r_search['adaptive']['recall']}")
    print(f"Adaptive Trace Steps: {len(r_search['adaptive']['trace'].get('steps', []))}")

    print("\nTesting /api/semantic/search...")
    req_sem_data = json.dumps({'query': 'how does skip list hierarchy work?', 'k': 2}).encode('utf-8')
    req_sem = urllib.request.Request('http://127.0.0.1:8000/api/semantic/search', data=req_sem_data, headers={'Content-Type': 'application/json'})
    r_sem = json.loads(urllib.request.urlopen(req_sem).read().decode())
    print(f"Top Hit: {r_sem['adaptive']['hits'][0]['title']} (Sim: {r_sem['adaptive']['hits'][0]['similarity']})")
    print(f"Memory Savings: {r_sem['comparison']['edge_savings_pct']}%")

    print("\nAll endpoints verified successfully!")

if __name__ == '__main__':
    test_api()
