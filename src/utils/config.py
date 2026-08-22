from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from src.data.dataset_recipe import DatasetRecipe

@dataclass
class TuningConfig:
    enabled: bool = False
    n_trials: int = 15
    sample_size: int = 12000
    val_sample_size: int = -1

@dataclass
class HyperparamsConfig:
    learning_rate: float = 2.5e-05
    epochs: int = 4
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    max_length: Optional[int] = None
    lambda_neg: float = 2.0
    w_doc: float = 1.0
    w_sent: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10

@dataclass
class ExperimentConfig:
    name: str
    model: str
    scope: str
    train_recipe: str
    dev_recipe: str
    eval_benchmarks: List[str] = field(default_factory=lambda: ['sentence'])
    seed: int = 42
    target_fpr: float = 0.01
    tuning: TuningConfig = field(default_factory=TuningConfig)
    hyperparams: HyperparamsConfig = field(default_factory=HyperparamsConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], global_seed: int = 42, global_target_fpr: float = 0.01) -> 'ExperimentConfig':
        model_name = data.get('model', 'svm').lower()
        scope = data.get('scope', 'sentence').lower()
        tune_data = data.get('tuning', {})
        if 'tune' in data:
            tune_data['enabled'] = bool(data['tune'])
        tuning_obj = TuningConfig(
            enabled=bool(tune_data.get('enabled', False)),
            n_trials=int(tune_data.get('n_trials', data.get('n_trials', 15))),
            sample_size=int(tune_data.get('sample_size', data.get('tuning_sample_size', 12000))),
            val_sample_size=int(tune_data.get('val_sample_size', data.get('val_sample_size', -1)))
        )
        hp_data = data.get('hyperparams', {})
        default_max_len = 128 if scope == 'sentence' else 256 if scope == 'mixed' else 384
        default_lr = 3e-05 if scope == 'sentence' else 2e-05
        hp_obj = HyperparamsConfig(
            learning_rate=float(hp_data.get('learning_rate', data.get('learning_rate', default_lr))),
            epochs=int(hp_data.get('epochs', data.get('epochs', 4))),
            batch_size=hp_data.get('batch_size', data.get('batch_size', None)),
            gradient_accumulation_steps=hp_data.get('gradient_accumulation_steps', data.get('gradient_accumulation_steps', None)),
            max_length=int(hp_data.get('max_length', data.get('max_length', default_max_len))),
            lambda_neg=float(hp_data.get('lambda_neg', data.get('lambda_neg', 2.0))),
            w_doc=float(hp_data.get('w_doc', data.get('w_doc', 1.0))),
            w_sent=float(hp_data.get('w_sent', data.get('w_sent', 1.0))),
            weight_decay=float(hp_data.get('weight_decay', data.get('weight_decay', 0.01))),
            warmup_ratio=float(hp_data.get('warmup_ratio', data.get('warmup_ratio', 0.10)))
        )
        eval_bms = data.get('eval_benchmarks', [scope])
        return cls(
            name=data.get('name', f'{model_name}_{scope}'),
            model=model_name,
            scope=scope,
            train_recipe=data['train_recipe'],
            dev_recipe=data['dev_recipe'],
            eval_benchmarks=eval_bms,
            seed=int(data.get('seed', global_seed)),
            target_fpr=float(data.get('target_fpr', global_target_fpr)),
            tuning=tuning_obj,
            hyperparams=hp_obj,
            raw_dict=data
        )

@dataclass
class GlobalExperimentConfig:
    seed: int
    target_fpr: float
    output_dir: str
    data_recipes: Dict[str, DatasetRecipe]
    experiments: List[ExperimentConfig]

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GlobalExperimentConfig':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Config file not found: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        g_seed = int(raw.get('seed', 42))
        g_target_fpr = float(raw.get('target_fpr', 0.01))
        g_out_dir = str(raw.get('output_dir', 'output'))
        recipes: Dict[str, DatasetRecipe] = {}
        for r_name, r_dict in raw.get('data_recipes', {}).items():
            recipes[r_name] = DatasetRecipe(
                name=r_name,
                splits=r_dict.get('splits', ['train']),
                include_full_abstracts=bool(r_dict.get('include_full_abstracts', False)),
                include_standard_sentences=bool(r_dict.get('include_standard_sentences', True)),
                include_full_abstract_sentences=bool(r_dict.get('include_full_abstract_sentences', False)),
                sample_size=int(r_dict.get('sample_size', -1)),
                seed=g_seed
            )
        exps: List[ExperimentConfig] = []
        for e_dict in raw.get('experiments', []):
            exps.append(ExperimentConfig.from_dict(e_dict, global_seed=g_seed, global_target_fpr=g_target_fpr))
        return cls(
            seed=g_seed,
            target_fpr=g_target_fpr,
            output_dir=g_out_dir,
            data_recipes=recipes,
            experiments=exps
        )