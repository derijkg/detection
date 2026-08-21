# src/utils/optuna_utils.py

import json
from pathlib import Path
from typing import Optional, Union
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from tqdm.auto import tqdm


class TqdmOptunaCallback:
    """
    Optuna callback that:
    1. Tracks optimization progress with a clean tqdm progress bar.
    2. Immediately saves `best_params.json` to disk whenever a new best score is reached.
    """
    def __init__(
        self, 
        n_trials: int, 
        desc: str = "Tuning Optuna", 
        save_path: Optional[Union[str, Path]] = None
    ):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.pbar = tqdm(total=n_trials, desc=desc, dynamic_ncols=True, leave=True)
        self.best_value: Optional[float] = None
        self.best_trial_num: Optional[int] = None
        self.save_path = Path(save_path) if save_path else None

    def __call__(self, study: Study, trial: FrozenTrial):
        self.pbar.update(1)

        if trial.state == TrialState.COMPLETE:
            is_maximize = (study.direction == optuna.study.StudyDirection.MAXIMIZE)
            if self.best_value is None:
                is_new_best = True
            elif is_maximize:
                is_new_best = trial.value > self.best_value
            else:
                is_new_best = trial.value < self.best_value

            if is_new_best:
                self.best_value = trial.value
                self.best_trial_num = trial.number

                # Dynamically write best parameters to disk immediately
                if self.save_path:
                    try:
                        self.save_path.parent.mkdir(parents=True, exist_ok=True)
                        payload = {
                            "_best_metric_value": float(self.best_value),
                            "_best_trial_number": int(self.best_trial_num),
                            **trial.params
                        }
                        self.save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    except Exception as e:
                        print(f"\n[Warning] Could not dynamically save best params: {e}")

            self.pbar.set_postfix({
                "Best": f"{self.best_value:.4f} (#{self.best_trial_num})",
                "Last": f"{trial.value:.4f}",
                "Status": "COMPLETE"
            })

        elif trial.state == TrialState.PRUNED:
            best_str = f"{self.best_value:.4f} (#{self.best_trial_num})" if self.best_value is not None else "N/A"
            self.pbar.set_postfix({
                "Best": best_str,
                "Last": "Pruned",
                "Status": "PRUNED"
            })

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()