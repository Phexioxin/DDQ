import random
import numpy as np
from deep_dialog.qlearning.hper_buffer import HierarchicalReplayBuffer

def run_demo(steps=1000, partitions=5):
    """Run a simple demonstration of HPER"""
    buf = HierarchicalReplayBuffer(steps, partitions)
    errors = []
    traj = []
    for i in xrange(steps):
        val = random.uniform(-1.0, 1.0)
        exp = (i, 0, val, i, 0, 0)
        buf.store(exp, val)
        batch, idxs, w = buf.sample(1)
        if idxs:
            traj.append(idxs[0][0])
        errors.append(abs(val))
    return errors, traj

def compare_baselines(runs=5):
    """Compare HPER with a random baseline using Welch's t-test"""
    hper_scores = []
    base_scores = []
    for i in xrange(runs):
        errors, _ = run_demo()
        hper_scores.append(np.mean(errors))
        base_scores.append(random.random())
    mean_a = np.mean(hper_scores)
    mean_b = np.mean(base_scores)
    var_a = np.var(hper_scores, ddof=1)
    var_b = np.var(base_scores, ddof=1)
    t = (mean_a - mean_b) / np.sqrt(var_a / len(hper_scores) + var_b / len(base_scores))
    return t

if __name__ == '__main__':
    errs, traj = run_demo(steps=100)
    print 'demo errors', np.mean(errs)
    print 't statistic', compare_baselines()
