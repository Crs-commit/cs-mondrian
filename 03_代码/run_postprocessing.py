"""Run the supported prediction-to-statistics pipeline into a new output directory.

Inputs are the four dataset directories from the private complete package.
No private output should be committed. Historical pre-strict comparisons,
PlanE and the original external backbone analyses are outside this runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if not data.is_dir():
        raise ValueError('Input directory does not exist')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; old results must not be reused')
    output.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    env = dict(os.environ, CS_MONDRIAN_DATA=str(data),
               CS_MONDRIAN_OUTPUT=str(output), PYTHONIOENCODING='utf-8',
               STRICT_CORE_RUN_LEVEL=str(output / 'strict_core_run_level.csv'),
               PYTHONHASHSEED='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    steps = [
        ('01_核心严格重算/strict_recompute.py', []),
        ('01_核心严格重算/generate_strict_report.py', []),
        ('02_Singh对照臂_新实验/singh_arm_recompute.py', []),
        ('02_Singh对照臂_新实验/fix_stats_eps_hum.py', []),
        ('02_Singh对照臂_新实验/singh_external_recompute.py', []),
        ('02_Singh对照臂_新实验/singh_self_check.py', []),
        ('03_替代调度消融_新实验/schedule_ablation_recompute.py', []),
        ('03_替代调度消融_新实验/schedule_ablation_external.py', []),
        ('03_替代调度消融_新实验/schedule_ablation_self_check.py', []),
        ('04_辅助脚本/capacity_extension.py', ['--mode', 'full']),
    ]
    import numpy, pandas, scipy
    manifest = {'python': sys.version, 'numpy': numpy.__version__,
                'pandas': pandas.__version__, 'scipy': scipy.__version__,
                'input_hashes': {}, 'steps': [],
                'scope': 'strict + Singh + schedule ablation + capacity; excludes PlanE and original external backbone pipeline'}
    for p in sorted(data.rglob('*')):
        if p.is_file() and p.suffix in ('.npz', '.csv'):
            manifest['input_hashes'][p.relative_to(data).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    logdir = output / 'execution_logs'
    logdir.mkdir()
    for name, flags in steps:
        start = time.monotonic()
        print('RUN', name, flush=True)
        with (logdir / (Path(name).stem + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, str(code / name), *flags],
                                    cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
        text = (logdir / (Path(name).stem + '.log')).read_text(encoding='utf-8')
        failed = result.returncode != 0 or '存在 FAIL' in text or '-> FAIL' in text
        manifest['steps'].append({'script': name, 'exit_code': result.returncode,
                                 'passed': not failed, 'seconds': round(time.monotonic()-start, 2),
                                 'sha256': hashlib.sha256((code/name).read_bytes()).hexdigest()})
        (output/'private_execution_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if failed:
            print(text[-5000:])
            raise SystemExit(1)
        print('PASS', name, flush=True)


if __name__ == '__main__':
    main()
