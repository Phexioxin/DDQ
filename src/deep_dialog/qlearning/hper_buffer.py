import numpy as np
import random

class SumTree(object):
    """A binary tree storing priorities for efficient sampling"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        parent = (idx - 1) / 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) / 2

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.n_entries


class PERBuffer(object):
    """Simple Prioritized Experience Replay without partitioning"""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0

    def _beta(self):
        beta = self.beta_start + (1.0 - self.beta_start) * float(self.frame) / self.beta_frames
        if beta > 1.0:
            beta = 1.0
        self.frame += 1
        return beta

    def store(self, transition, td_error=None):
        if td_error is None:
            p = self.max_priority
        else:
            p = (abs(td_error) + 1e-6) ** self.alpha
            if p > self.max_priority:
                self.max_priority = p
        self.tree.add(p, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        probs = []
        total = self.tree.total()
        n = len(self.tree)
        beta = self._beta()
        for i in xrange(min(batch_size, n)):
            s = random.random() * total
            idx, p, data = self.tree.get(s)
            prob = p / total
            batch.append(data)
            idxs.append(idx)
            probs.append(prob)
        weights = np.array([(n * p) ** (-beta) for p in probs])
        if len(weights) > 0:
            weights = weights / np.max(weights)
        return batch, idxs, weights

    def update(self, idxs, errors):
        for i in xrange(len(idxs)):
            p = (abs(errors[i]) + 1e-6) ** self.alpha
            if p > self.max_priority:
                self.max_priority = p
            self.tree.update(idxs[i], p)

    def __len__(self):
        return len(self.tree)


class HierarchicalReplayBuffer(object):
    """Hierarchical Prioritized Experience Replay Buffer.

    Samples are partitioned by experience source (real/sim) and trajectory
    length (short/med/long). Each partition maintains its own Sum-Tree for
    priority based sampling. Two-level quotas over source and length are used
    to allocate batch slots. When rotation is enabled each partition is assumed
    to be selected with probability ``1/partitions`` and the overall sample
    probability becomes ``p/(tree.total()*partitions)``. The associated IS
    weights follow ``w_i=(1/N*1/P(i))^beta``.
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4,
                 beta_increment_per_sampling=0.001, disable_rotation=0,
                 disable_priority=0, len_th=(6, 12)):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.disable_rotation = disable_rotation
        self.disable_priority = disable_priority
        self.len_th = len_th
        self.max_priority = 1.0

        self.partitions = 6  # 2 sources * 3 length buckets
        per_capacity = max(1, capacity / self.partitions)
        self.trees = [SumTree(per_capacity) for _ in xrange(self.partitions)]

    def _partition_index(self, src, turns):
        if src == 'sim':
            s = 1
        else:
            s = 0
        if turns < self.len_th[0]:
            l = 0
        elif turns <= self.len_th[1]:
            l = 1
        else:
            l = 2
        return s * 3 + l

    def store(self, data, meta):
        part_idx = self._partition_index(meta.get('src', 'real'),
                                         meta.get('turns', 0))
        if self.disable_priority:
            priority = 1.0
        else:
            priority = self.max_priority
        self.trees[part_idx].add(priority, data)

    def sample(self, n, quota_src='1:1', quota_len='1:1:1'):
        batch = []
        idxs = []
        probs = []
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # compute desired quota for each partition
        a, b = [float(x) for x in quota_src.split(':')]
        x, y, z = [float(x) for x in quota_len.split(':')]
        src_ratio = [a/(a+b), b/(a+b)]
        len_ratio = [x/(x+y+z), y/(x+y+z), z/(x+y+z)]
        desired = []
        for s in xrange(2):
            for l in xrange(3):
                desired.append(int(round(n * src_ratio[s] * len_ratio[l])))
        diff = n - sum(desired)
        while diff != 0:
            idx = np.argmax([len(self.trees[i]) for i in xrange(self.partitions)])
            desired[idx] += 1 if diff > 0 else -1
            diff = n - sum(desired)

        total_len = len(self)
        for part_idx in xrange(self.partitions):
            tree = self.trees[part_idx]
            need = desired[part_idx]
            for _ in xrange(need):
                if len(tree) == 0:
                    continue
                if self.disable_priority:
                    leaf = random.randint(0, tree.n_entries - 1)
                    idx = leaf + tree.capacity - 1
                    p = tree.tree[idx]
                    data = tree.data[leaf]
                else:
                    s = random.random() * tree.total()
                    idx, p, data = tree.get(s)
                batch.append(data)
                idxs.append((part_idx, idx))
                if self.disable_rotation:
                    # Without round-robin the probability of selecting a
                    # partition is ``need / n``. The sample probability is
                    # ``(need/n) * (p / tree.total())`` which is used later
                    # for IS weight computation.
                    part_prob = float(need) / float(n) if n > 0 else 0
                    denom = tree.total()
                    prob = part_prob * (p / denom if denom > 0 else 0)
                else:
                    # Under round-robin each partition is chosen with
                    # probability ``1/partitions``. The probability of a
                    # leaf becomes ``p/(tree.total()*partitions)``.
                    prob = p / (tree.total() * self.partitions)
                probs.append(prob)

        while len(batch) < n and len(batch) > 0:
            k = random.randint(0, len(batch) - 1)
            batch.append(batch[k])
            idxs.append(idxs[k])
            probs.append(probs[k])

        weights = np.array([(total_len * max(p, 1e-10)) ** (-self.beta)
                            for p in probs])
        if len(weights) > 0:
            cutoff = np.percentile(weights, 99)
            weights = np.minimum(weights, cutoff)
            weights = weights / np.max(weights)
        return batch, idxs, weights

    def update(self, idxs, errors):
        for i in xrange(len(idxs)):
            part_idx, tree_idx = idxs[i]
            tree = self.trees[part_idx]
            p = (abs(errors[i]) + 1e-6) ** self.alpha
            tree.update(tree_idx, p)
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        total = 0
        for tree in self.trees:
            total += len(tree)
        return total
