import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml

def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return commit
    except Exception:
        return 'unknown'

class RunContext:

    def __init__(self, run_id: str, run_dir: Path, manifest_path: Path, model_name: str, scope: str, exp_name: str):
        self.run_id = run_id
        self.run_dir = run_dir
        self.manifest_path = manifest_path
        self.model_name = model_name
        self.scope = scope
        self.exp_name = exp_name
        self.start_time = time.time()
        self.model_dir = self.run_dir / 'model'
        self.predictions_dir = self.run_dir / 'predictions'
        self.metrics_dir = self.run_dir / 'metrics'
        self.plots_dir = self.run_dir / 'plots'
        for d in [self.run_dir, self.model_dir, self.predictions_dir, self.metrics_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_run_metadata(self, config: Dict[str, Any], train_recipe_meta: Optional[Dict[str, Any]]=None, dev_recipe_meta: Optional[Dict[str, Any]]=None):
        config_snapshot_file = self.run_dir / 'config_snapshot.yaml'
        with open(config_snapshot_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        meta = {
            'run_id': self.run_id,
            'experiment_name': self.exp_name,
            'model': self.model_name,
            'scope': self.scope,
            'timestamp': datetime.datetime.now().isoformat(),
            'git_commit': get_git_commit(),
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'train_recipe': train_recipe_meta or {},
            'dev_recipe': dev_recipe_meta or {}
        }
        (self.run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    def record_to_manifest(self, summary_dict: Optional[Dict[str, Any]]=None, status: str='COMPLETED', error_msg: Optional[str]=None):
        duration_sec = round(time.time() - self.start_time, 2)
        summary = summary_dict or {}
        record = {
            'run_id': self.run_id,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'name': self.exp_name,
            'model': self.model_name,
            'scope': self.scope,
            'status': status,
            'duration_sec': duration_sec,
            'git_commit': get_git_commit(),
            'calibrated_threshold': summary.get('calibrated_threshold'),
            'overall_pauc': summary.get('overall_pauc'),
            'tpr_at_1fpr': summary.get('tpr_at_1fpr'),
            'overall_roc_auc': summary.get('overall_roc_auc'),
            'f1_ai': summary.get('f1_ai'),
            'fpr_human': summary.get('fpr_human'),
            'brier_score': summary.get('brier_score'),
            'overall_mcc': summary.get('overall_mcc'),
            'error_msg': error_msg,
            'run_dir': str(self.run_dir.resolve())
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

class RunTracker:

    def __init__(self, output_root: Union[str, Path]='output'):
        self.output_root = Path(output_root)
        self.manifest_path = self.output_root / 'manifest.jsonl'
        self.runs_root = self.output_root / 'runs'

    def start_run(self, exp_name: str, model_name: str, scope: str, config: Dict[str, Any], train_recipe_meta: Optional[Dict[str, Any]]=None, dev_recipe_meta: Optional[Dict[str, Any]]=None) -> RunContext:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f'EXP_{timestamp}_{model_name}_{scope}_{exp_name}'
        run_dir = self.runs_root / run_id
        ctx = RunContext(run_id=run_id, run_dir=run_dir, manifest_path=self.manifest_path, model_name=model_name, scope=scope, exp_name=exp_name)
        ctx.save_run_metadata(config=config, train_recipe_meta=train_recipe_meta, dev_recipe_meta=dev_recipe_meta)
        return ctx

    def load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame()
        records = []
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return pd.DataFrame(records)

def show_leaderboard(output_root: str='output', scope: Optional[str]=None):
    tracker = RunTracker(output_root=output_root)
    df = tracker.load_manifest()
    if df.empty:
        print(f'No runs recorded in {tracker.manifest_path}')
        return
    if scope:
        df = df[df['scope'] == scope]
    cols_to_show = ['timestamp', 'run_id', 'model', 'scope', 'status', 'overall_pauc', 'tpr_at_1fpr', 'f1_ai', 'fpr_human']
    avail_cols = [c for c in cols_to_show if c in df.columns]
    display_df = df[avail_cols].copy()
    if 'overall_pauc' in display_df:
        display_df['overall_pauc'] = display_df['overall_pauc'].apply(lambda v: f'{v:.4f}' if pd.notna(v) and isinstance(v, (int, float)) else '-')
    if 'tpr_at_1fpr' in display_df:
        display_df['tpr_at_1fpr'] = display_df['tpr_at_1fpr'].apply(lambda v: f'{v * 100:.2f}%' if pd.notna(v) and isinstance(v, (int, float)) else '-')
    if 'f1_ai' in display_df:
        display_df['f1_ai'] = display_df['f1_ai'].apply(lambda v: f'{v:.4f}' if pd.notna(v) and isinstance(v, (int, float)) else '-')
    if 'fpr_human' in display_df:
        display_df['fpr_human'] = display_df['fpr_human'].apply(lambda v: f'{v * 100:.2f}%' if pd.notna(v) and isinstance(v, (int, float)) else '-')
    print('\n' + '=' * 95)
    print('                              EXPERIMENT RUN LEADERBOARD')
    print('=' * 95)
    print(display_df.to_string(index=False))
    print('=' * 95 + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='View Experiment Manifest')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--scope', type=str, default=None, help='Filter by scope (e.g. sentence, full)')
    args = parser.parse_args()
    show_leaderboard(output_root=args.output_dir, scope=args.scope)