import os
import sys
import subprocess
import json
import csv
import platform
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_DIR = os.path.join(ROOT_DIR, 'cpp')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXE_PATH = os.path.join(CPP_DIR, 'benchmark_runner.exe')

def get_system_info():
    info = {
        'os': platform.platform(),
        'python_version': platform.python_version(),
        'cpu': platform.processor(),
        'architecture': platform.machine(),
    }
    try:
        if platform.system() == 'Windows':
            out = subprocess.check_output(['powershell', '-Command', 'Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name'], text=True)
            info['cpu_model'] = out.strip()
            ram = subprocess.check_output(['powershell', '-Command', '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB'], text=True)
            info['ram_gb'] = str(round(float(ram.strip()), 1)) + ' GB'
    except Exception:
        pass
    return info

def ensure_cpp_compiled():
    print('[*] Compiling C++ benchmark engine with AVX2 & FMA...')
    src = os.path.join(CPP_DIR, 'benchmark_main.cpp')
    cmd = ['g++', '-O3', '-mavx2', '-mfma', '-std=c++17', src, '-o', EXE_PATH]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print('[-] Compilation failed!')
        sys.exit(1)
    print('[+] Compilation successful: ' + EXE_PATH)

def run_benchmarks():
    print('=' * 85)
    print('          AdaptiveVec Research Paper Reproduction & Benchmark Suite          ')
    print('=' * 85)
    sys_info = get_system_info()
    cpu_desc = sys_info.get('cpu_model', sys_info.get('cpu', 'Intel x86_64'))
    ram_desc = sys_info.get('ram_gb', 'N/A')
    print('Testbed Hardware : ' + cpu_desc + ' (' + ram_desc + ')')
    print('OS & Platform    : ' + sys_info['os'])
    print('Compiler         : g++ (GCC with -O3 -mavx2 -mfma -std=c++17)')
    print('=' * 85 + '\\n')
    
    ensure_cpp_compiled()
    
    # 1. Real SIFT-100K dataset
    sift_100k = os.path.join(DATA_DIR, 'sift_subset_100k.fvecs')
    sift_query = os.path.join(DATA_DIR, 'sift_query.fvecs')
    sift_gt = os.path.join(DATA_DIR, 'sift_subset_groundtruth.ivecs')
    sift_json = os.path.join(DATA_DIR, 'sift_results.json')
    sift_csv = os.path.join(DATA_DIR, 'sift_results.csv')
    
    if not (os.path.exists(sift_100k) and os.path.exists(sift_query) and os.path.exists(sift_gt)):
        print('\n[*] Note: Canonical SIFT dataset not found in data/. Preparing from Texmex IRISA source...')
        prep_script = os.path.join(ROOT_DIR, 'benchmarks', 'prepare_datasets.py')
        subprocess.run([sys.executable, prep_script])
        
    if os.path.exists(sift_100k) and os.path.exists(sift_query) and os.path.exists(sift_gt):
        print('\n>>> [Benchmark 1/2] 6-Step Ablation on Genuine SIFT-100K subset (10,000 queries) <<<')
        cmd = [
            EXE_PATH,
            '--dataset', sift_100k,
            '--queries', sift_query,
            '--gt', sift_gt,
            '--label', 'SIFT-100K subset',
            '--max_samples', '100000',
            '--max_queries', '10000',
            '--ef_search', '64',
            '--out', sift_json,
            '--csv', sift_csv
        ]
        subprocess.run(cmd)
    
    # 2. Synthetic Multi-Cluster benchmark
    synth_json = os.path.join(DATA_DIR, 'synthetic_results.json')
    synth_csv = os.path.join(DATA_DIR, 'synthetic_results.csv')
    print('\n>>> [Benchmark 2/2] Synthetic Multi-Cluster Manifold (50,000 vectors, 8 clusters) <<<')
    subprocess.run([
        EXE_PATH,
        '--synthetic',
        '--max_samples', '50000',
        '--max_queries', '1000',
        '--ef_search', '64',
        '--label', 'Synthetic-Multi-Cluster',
        '--out', synth_json,
        '--csv', synth_csv
    ])
    
    # 3. Consolidate into root benchmark_results.json and benchmark_results.csv
    consolidated_records = []
    for jpath in [sift_json, synth_json]:
        if os.path.exists(jpath):
            with open(jpath, 'r') as f:
                data = json.load(f)
                consolidated_records.extend(data.get('benchmark_results', []))
                
    root_json = os.path.join(ROOT_DIR, 'benchmark_results.json')
    root_csv = os.path.join(ROOT_DIR, 'benchmark_results.csv')
    
    with open(root_json, 'w') as f:
        json.dump({'benchmark_results': consolidated_records}, f, indent=2)
        
    if consolidated_records:
        keys = list(consolidated_records[0].keys())
        with open(root_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in consolidated_records:
                writer.writerow(r)
                
    # 4. Report canonical dataset provenance and DBpedia status
    print('\n' + '=' * 85)
    print('  SIFT-100K Dataset   : Extracted from canonical Texmex IRISA (ftp://ftp.irisa.fr)')
    print('  GloVe-100 Dataset   : Canonical source available from Stanford NLP (nlp.stanford.edu)')
    print('  DBpedia-100K Status : NOT RUN (Real OpenAI text-embedding-3-small vectors not available)')
    print('=' * 85)
    print('\n[+] Consolidated empirical benchmark results written to:')
    print('    - ' + root_json)
    print('    - ' + root_csv)

if __name__ == '__main__':
    run_benchmarks()
