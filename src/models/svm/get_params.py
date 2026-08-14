# detection/scripts/extract_pickle_params.py

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC


# =============================================================================
# RECURSIVE PIPELINE INSPECTION HELPERS
# =============================================================================

def find_instances(obj: Any, target_class: type) -> List[Any]:
    """Recursively traverses pipelines, feature unions, and dicts to find all instances of target_class."""
    instances = []
    if isinstance(obj, target_class):
        instances.append(obj)
    elif isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            instances.extend(find_instances(step, target_class))
    elif isinstance(obj, FeatureUnion):
        for name, trans in obj.transformer_list:
            instances.extend(find_instances(trans, target_class))
    elif hasattr(obj, 'transformer_list'):
        for name, trans in obj.transformer_list:
            instances.extend(find_instances(trans, target_class))
    elif hasattr(obj, 'named_steps'):
        for name, step in obj.named_steps.items():
            instances.extend(find_instances(step, target_class))
    elif isinstance(obj, dict):
        for key, val in obj.items():
            instances.extend(find_instances(val, target_class))
    return instances


def sanitize_value(val: Any) -> Any:
    """Converts non-JSON serializable numpy types and functions to clean primitives."""
    if val is None:
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [sanitize_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): sanitize_value(v) for k, v in val.items()}
    if callable(val):
        return val.__name__
    return str(val)


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def extract_tfidf_parameters(pipeline_or_obj: Any) -> Dict[str, Any]:
    """Extracts all word and char TF-IDF vectorizer parameters and learned vocab sizes."""
    tfidf_vectorizers = find_instances(pipeline_or_obj, TfidfVectorizer)
    tfidf_data = {}

    for idx, vec in enumerate(tfidf_vectorizers):
        params = vec.get_params()
        analyzer = params.get('analyzer', 'word')
        name = f"tfidf_{analyzer}_{idx}" if len(tfidf_vectorizers) > 1 else f"tfidf_{analyzer}"

        clean_params = {}
        for k, v in params.items():
            if k == 'stop_words':
                clean_params['stop_words_count'] = len(v) if hasattr(v, '__len__') else (1 if v else 0)
            else:
                clean_params[k] = sanitize_value(v)

        # Learned vocabulary size
        if hasattr(vec, 'vocabulary_'):
            clean_params['learned_vocabulary_size'] = len(vec.vocabulary_)

        tfidf_data[name] = clean_params

    return tfidf_data


def extract_classifier_parameters(pipeline_or_obj: Any) -> Dict[str, Any]:
    """Extracts model hyperparameters, calibration state, and learned weights/coefs."""
    clf_instances = find_instances(pipeline_or_obj, (LinearSVC, SVC, CalibratedClassifierCV))
    
    if not clf_instances:
        # Fallback: check dict or named_steps directly
        if isinstance(pipeline_or_obj, Pipeline) and 'classifier' in pipeline_or_obj.named_steps:
            clf_instances = [pipeline_or_obj.named_steps['classifier']]

    if not clf_instances:
        return {"error": "No classifier found in pickle file"}

    clf = clf_instances[-1]  # Take main classifier
    clf_data = {
        "class_name": clf.__class__.__name__,
        "hyperparameters": sanitize_value(clf.get_params())
    }

    base_est = clf

    # Handle CalibratedClassifierCV wrapper
    if isinstance(clf, CalibratedClassifierCV):
        clf_data["calibration_method"] = clf.method
        if hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
            clf_data["num_calibrated_folds"] = len(clf.calibrated_classifiers_)
            first_fold = clf.calibrated_classifiers_[0]
            base_est = getattr(first_fold, "estimator", getattr(first_fold, "base_estimator", None))

    # Extract learned weights / coefficients
    if base_est is not None:
        clf_data["base_estimator_class"] = base_est.__class__.__name__
        if hasattr(base_est, "coef_") and base_est.coef_ is not None:
            coef = base_est.coef_.toarray() if hasattr(base_est.coef_, "toarray") else base_est.coef_
            clf_data["learned_weights"] = {
                "shape": list(coef.shape),
                "min_weight": float(np.min(coef)),
                "max_weight": float(np.max(coef)),
                "mean_weight": float(np.mean(coef)),
                "std_weight": float(np.std(coef)),
            }
        if hasattr(base_est, "intercept_") and base_est.intercept_ is not None:
            clf_data["learned_intercept"] = sanitize_value(base_est.intercept_)

    return clf_data


def inspect_pickle_file(file_path: str) -> Dict[str, Any]:
    """Inspects a single pickle file and extracts all metadata, TF-IDF, and model parameters."""
    print(f"\n==================================================")
    print(f" LOADING PICKLE FILE: {file_path}")
    print(f"==================================================")

    obj = joblib.load(file_path)

    report = {
        "file_name": os.path.basename(file_path),
        "file_path": str(Path(file_path).resolve()),
        "root_object_type": obj.__class__.__name__
    }

    # Extract Metadata if attached
    if isinstance(obj, dict) and "metadata" in obj:
        report["metadata"] = sanitize_value(obj["metadata"])
    elif hasattr(obj, "metadata"):
        report["metadata"] = sanitize_value(getattr(obj, "metadata"))
    elif hasattr(obj, "optimal_threshold"):
        report["metadata"] = {"optimal_threshold": getattr(obj, "optimal_threshold")}

    # 1. Extract TF-IDF Parameters
    report["tfidf_parameters"] = extract_tfidf_parameters(obj)

    # 2. Extract Classifier Parameters
    report["classifier_parameters"] = extract_classifier_parameters(obj)

    return report


def convert_to_yaml_config(report: Dict[str, Any]) -> Dict[str, Any]:
    """Translates extracted parameters into a ready-to-use YAML configuration for your repo."""
    tfidf = report.get("tfidf_parameters", {})
    clf = report.get("classifier_parameters", {})
    clf_params = clf.get("hyperparameters", {})

    # Extract word and char params
    word_params = next((v for k, v in tfidf.items() if "word" in k), {})
    char_params = next((v for k, v in tfidf.items() if "char" in k), {})

    yaml_config = {
        "model": {
            "name": "svm",
            "granularity": report.get("metadata", {}).get("granularity", "full"),
            "use_stylometrics": report.get("metadata", {}).get("use_stylometrics", True),
            "calibrate": "CalibratedClassifierCV" in clf.get("class_name", "")
        },
        "training": {
            "output_dir": "outputs/checkpoints/svm",
            "kernel": clf_params.get("kernel", "linear"),
            "C": clf_params.get("C", 1.0),
            "class_weight": clf_params.get("class_weight", "balanced"),
            "word_min_ngram": word_params.get("ngram_range", [1, 3])[0] if word_params.get("ngram_range") else 1,
            "word_max_ngram": word_params.get("ngram_range", [1, 3])[1] if word_params.get("ngram_range") else 3,
            "word_max_features": word_params.get("max_features", 50000),
            "word_min_df": word_params.get("min_df", 2),
            "word_max_df": word_params.get("max_df", 0.95),
            "char_min_ngram": char_params.get("ngram_range", [3, 5])[0] if char_params.get("ngram_range") else 3,
            "char_max_ngram": char_params.get("ngram_range", [3, 5])[1] if char_params.get("ngram_range") else 5,
            "char_max_features": char_params.get("max_features", 50000),
            "char_min_df": char_params.get("min_df", 2),
            "char_max_df": char_params.get("max_df", 0.95),
        }
    }
    return yaml_config


# =============================================================================
# MAIN CLI ROUTINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract all TF-IDF and Model parameters from trained pickle files.")
    parser.add_argument("--path", type=str, required=True, help="Path to .pkl or .joblib file (or folder containing pickle files)")
    parser.add_argument("--out_json", type=str, default=None, help="Path to save output JSON (default: extracted_params_<filename>.json)")
    args = parser.parse_args()

    target_path = Path(args.path)

    if target_path.is_dir():
        pickle_files = list(target_path.glob("*.pkl")) + list(target_path.glob("*.joblib"))
        if not pickle_files:
            print(f"No .pkl or .joblib files found in directory: {target_path}")
            return
    else:
        pickle_files = [target_path]

    for pkl_file in pickle_files:
        report = inspect_pickle_file(str(pkl_file))
        
        # Pretty print report to console
        print("\n--- EXTRACTED PARAMETERS SUMMARY ---")
        print(json.dumps(report, indent=4))

        # Generate YAML config format snippet
        yaml_equivalent = convert_to_yaml_config(report)
        print("\n--- READY-TO-USE YAML CONFIG EQUIVALENT ---")
        print(json.dumps(yaml_equivalent, indent=4))

        # Save to JSON
        out_json_path = args.out_json or f"extracted_params_{pkl_file.stem}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"extracted_report": report, "yaml_config_equivalent": yaml_equivalent}, f, indent=4)
        print(f"\n[SAVED] Parameter report exported to: '{out_json_path}'")


if __name__ == "__main__":
    main()