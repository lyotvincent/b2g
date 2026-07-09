"""
Utilities for selecting the best grouping mode.
"""

import copy
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd


ModeEvaluator = Callable[[Dict[str, Dict[str, Any]]], Any]


def _parse_evaluator_output(result: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(result, str):
        return result, {}
    if isinstance(result, tuple) and len(result) == 2:
        best_mode, payload = result
        return best_mode, dict(payload or {})
    if isinstance(result, dict):
        if "best_mode" not in result:
            raise ValueError("mode_evaluator result dict must contain 'best_mode'")
        payload = dict(result)
        best_mode = payload.pop("best_mode")
        return best_mode, payload
    raise TypeError(
        "mode_evaluator must return either a mode string, a (mode, payload) tuple, "
        "or a dict containing 'best_mode'"
    )


def _sanitize_uns_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_uns_value(sub_value) for key, sub_value in value.items()}

    if isinstance(value, list):
        if len(value) == 0:
            return []
        if all(isinstance(item, dict) for item in value):
            return pd.DataFrame([{str(k): _sanitize_uns_value(v) for k, v in item.items()} for item in value])
        return [_sanitize_uns_value(item) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_uns_value(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    return value


def _extract_batch_to_group_mapping(grouped, group_key: str, batch_key: str) -> Dict[str, str]:
    if batch_key not in grouped.obs.columns or group_key not in grouped.obs.columns:
        return {}

    batch_series = grouped.obs[batch_key].astype(str)
    group_series = grouped.obs[group_key].astype(str)
    mapping_df = pd.DataFrame({"batch": batch_series, "group": group_series}).drop_duplicates()
    return dict(zip(mapping_df["batch"], mapping_df["group"]))


def select_best_grouping_mode(adata, config, key_added="groups_metacell_adaptive"):
    from .grouping import group_batches

    evaluator = getattr(config, "mode_evaluator", None)
    if evaluator is None or not callable(evaluator):
        raise ValueError(
            "grouping_mode='auto' requires config.mode_evaluator to be a callable. "
            "Use a dedicated ARMS comparison script to build this evaluator and pass it "
            "through b2g.group(..., mode='auto', mode_evaluator=callable)."
        )

    candidate_modes = tuple(getattr(config, "mode_candidates", ("tree", "split", "prior")))
    candidate_results = {}
    errors = {}

    for mode in candidate_modes:
        candidate_config = copy.deepcopy(config)
        candidate_config.grouping_mode = mode
        candidate_key = f"{key_added}_{mode}"
        try:
            grouped = group_batches(adata.copy(), candidate_config, key_added=candidate_key)
            batch_key = candidate_config.column_mapping["batch"]
            candidate_results[mode] = {
                "adata": grouped,
                "key_added": candidate_key,
                "config": candidate_config,
                "batch_to_group": _extract_batch_to_group_mapping(grouped, candidate_key, batch_key),
            }
        except Exception as exc:
            errors[mode] = str(exc)

    if not candidate_results:
        raise ValueError(f"All candidate grouping modes failed: {errors}")

    best_mode, payload = _parse_evaluator_output(evaluator(candidate_results))
    if best_mode not in candidate_results:
        raise ValueError(f"mode_evaluator selected unknown mode: {best_mode}")

    selected = candidate_results[best_mode]["adata"]
    selected_key = candidate_results[best_mode]["key_added"]
    if selected_key != key_added:
        selected.obs[key_added] = selected.obs[selected_key].copy()

    for mode, result in candidate_results.items():
        mode_key = result["key_added"]
        obs_column = f"b2g_group_{mode}"
        if mode_key in result["adata"].obs.columns:
            selected.obs[obs_column] = result["adata"].obs[mode_key].reindex(selected.obs_names)
        selected.uns[f"b2g_batch_to_group_{mode}"] = result.get("batch_to_group", {})

    selected.uns["b2g_grouping_mode"] = best_mode
    payload.setdefault("best_mode", best_mode)
    payload.setdefault("failed_modes", errors)
    selected.uns["b2g_mode_selection"] = _sanitize_uns_value(payload)
    return selected
