import torch
from twosample import binom_test
import numpy as np

def fast_certify(samples_flattened, n0, n, tau=0.75, alpha=0.001, stats=None, non_ignore_idx=None):
    assert samples_flattened.shape[0] == n
    if not isinstance(samples_flattened, torch.Tensor):
        samples_flattened = torch.from_numpy(samples_flattened)
    if non_ignore_idx is None:
        non_ignore_idx = [True]*samples_flattened.shape[1]
    modes, _ = torch.mode(samples_flattened[:n0], 0)
    counts_at_modes = (samples_flattened[n0:] == modes.unsqueeze(0)).sum(0)
    pvals_ = binom_test(np.array(counts_at_modes), np.array([n-n0]*len(samples_flattened[0])), np.array([tau]*len(samples_flattened[0])), alt='greater')
    abstain = pvals_ > alpha/len(samples_flattened[0])
    modes = modes.cpu().numpy()    
    modes[np.array(abstain)] = -1
    if stats is not None:
        d = {}
        d['fluctuations'] = samples_flattened[:, abstain & non_ignore_idx].cpu().numpy()
        return modes, d
    return torch.from_numpy(modes)