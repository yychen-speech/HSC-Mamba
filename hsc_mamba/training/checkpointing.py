"""Select and average checkpoints using validation scores only."""
import math
import os
import re

import torch


VALID_PATTERN = re.compile(
    r'^best_valid_epoch(?P<epoch>\d+)_score'
    r'(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\.ckpt$'
)


def select_validation_checkpoints(paths, count=5):
    ranked = []
    for path in paths:
        match = VALID_PATTERN.fullmatch(os.path.basename(path))
        if match and math.isfinite(float(match['score'])):
            ranked.append((float(match['score']), int(match['epoch']), path))
    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    if len(ranked) < count:
        raise RuntimeError(f'Need {count} validation-ranked checkpoints; found {len(ranked)}. '
                           'Test-ranked or last checkpoints cannot substitute for them.')
    return [entry[2] for entry in ranked[:count]]


def average_validation_checkpoints(paths, count=5):
    selected = select_validation_checkpoints(paths, count)
    result = None
    original_dtypes = {}
    for path in selected:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        # Validation evaluates EMA, so average those same model parameters.
        state = checkpoint.get('ema_state_dict')
        if state is None:
            raise RuntimeError(f'Missing validation EMA state: {path}')
        state = {key: value for key, value in state.items()
                 if not key.endswith(('total_ops', 'total_params'))}
        if result is None:
            result = {}
            for key, value in state.items():
                original_dtypes[key] = value.dtype
                result[key] = value.to(torch.complex128 if value.is_complex() else torch.float64).clone() \
                    if value.is_floating_point() or value.is_complex() else value.clone()
        else:
            if result.keys() != state.keys():
                raise RuntimeError(f'Incompatible checkpoint keys: {path}')
            for key, value in state.items():
                if result[key].shape != value.shape or original_dtypes[key] != value.dtype:
                    raise RuntimeError(f'Incompatible tensor {key}: {path}')
                if value.is_floating_point() or value.is_complex():
                    result[key].add_(value)
                elif not torch.equal(result[key], value):
                    raise RuntimeError(f'Non-floating buffer differs: {key}')
    for key in result:
        if result[key].is_floating_point() or result[key].is_complex():
            result[key] = (result[key] / count).to(original_dtypes[key])
        if (result[key].is_floating_point() or result[key].is_complex()) and not torch.isfinite(result[key]).all():
            raise RuntimeError(f'Non-finite averaged parameter: {key}')
    return result, selected
