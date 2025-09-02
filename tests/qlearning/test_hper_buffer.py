import builtins
builtins.xrange = range
import numpy as np

from deep_dialog.qlearning.hper_buffer import HierarchicalReplayBuffer


def make_transition(idx):
    return ('s%r' % idx, idx, idx, 's%r' % (idx+1), False)


def test_store_and_sample_quota():
    buf = HierarchicalReplayBuffer(100, disable_priority=1)
    for i in range(10):
        meta = {'src': 'real', 'turns': i % 15}
        buf.store(make_transition(i), meta)
    for i in range(10):
        meta = {'src': 'sim', 'turns': i % 15}
        buf.store(make_transition(100+i), meta)
    batch, idxs, weights = buf.sample(8, '1:1', '1:1:1')
    assert len(batch) == 8
    assert len(idxs) == 8
    assert len(weights) == 8
    assert np.all(weights <= 1.0)
