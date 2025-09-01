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
    """Hierarchical Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, partitions=5, alpha=0.6, beta=0.4,
                 beta_increment_per_sampling=0.001, disable_rotation=0,
                 disable_priority=0):
        self.capacity = capacity
        self.partitions = partitions
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.disable_rotation = disable_rotation
        self.disable_priority = disable_priority
        self.max_priority = 1.0

        per_capacity = max(1, capacity / partitions)
        self.trees = []
        for _ in xrange(partitions):
            self.trees.append(SumTree(per_capacity))
        self.meta_tree = SumTree(partitions)

        self.min_val = -1.0
        self.max_val = 1.0
        self._recompute_boundaries()
        self.next_partition = 0

    def _recompute_boundaries(self):
        self.boundaries = np.linspace(self.min_val, self.max_val,
                                      self.partitions + 1).tolist()

    def _get_partition(self, value):
        if value < self.min_val:
            self.min_val = value
            self._recompute_boundaries()
        elif value > self.max_val:
            self.max_val = value
            self._recompute_boundaries()
        for i in xrange(self.partitions):
            if value <= self.boundaries[i + 1]:
                return i
        return self.partitions - 1

    def _update_meta(self, part_idx):
        if self.disable_priority:
            total = len(self.trees[part_idx])
        else:
            total = self.trees[part_idx].total()
        self.meta_tree.update(part_idx + self.meta_tree.capacity - 1, total)

    def store(self, data, value):
        """Insert a transition with associated state value.

        The state value decides which partition will hold the sample. New
        entries are always inserted with maximum priority so that they can be
        seen at least once before being updated by TD errors.
        """
        part_idx = self._get_partition(value)
        if self.disable_priority:
            priority = 1.0
        else:
            priority = self.max_priority
        self.trees[part_idx].add(priority, data)
        self._update_meta(part_idx)

    def sample(self, n):
        """Sample ``n`` items, returning data, indices and IS weights."""

        batch = []
        idxs = []
        weights = []
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        total_meta = self.meta_tree.total()
        total_len = len(self)
        i = 0
        while i < n and total_len > 0:
            if self.disable_rotation:
                s = random.random() * total_meta
                part_idx, _, _ = self.meta_tree.get(s)
                part_idx = part_idx - self.meta_tree.capacity + 1
            else:
                part_idx = self.next_partition
                self.next_partition = (self.next_partition + 1) % self.partitions
            tree = self.trees[part_idx]
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
                if total_meta == 0:
                    prob = 1.0
                else:
                    prob = p / total_meta
            else:
                # Round-robin partition selection: P(partition)=1/partitions.
                # Conditional probability inside partition follows
                # PER definition P(i|k)=p / tree.total().
                # Overall probability P(i)=P(k) * P(i|k)=p/(tree.total()*partitions)
                prob = p / (tree.total() * self.partitions)
            weight = (total_len * prob) ** (-self.beta)
            weights.append(weight)
            i += 1
        weights = np.array(weights)
        if len(weights) > 0:
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
            self._update_meta(part_idx)

    def __len__(self):
        total = 0
        for tree in self.trees:
            total += len(tree)
        return total
