import math
import random
import mmap
import struct
import os
import argparse
import json
from pathlib import Path
import time

class BanditEnv:
    def __init__(self, memory_limit, config_path):
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.memory_limit = memory_limit
        self.io_speed     = config["io_speed"]
        self.layer_sizes  = config["layer_sizes"]
        self.t_compute    = config["t_compute"]
        self.t_release    = config["t_release"]
        self.total_layers = len(self.layer_sizes)
        self.t_load       = [sz / self.io_speed * 1000 for sz in self.layer_sizes]
        print("total layers:", self.total_layers, "total size:", sum(self.layer_sizes))
        print("io speed:", self.io_speed)
        print("t_load:", self.t_load)

        assert len(self.t_compute) == self.total_layers, "Mismatch: t_compute vs total_layers"
        assert len(self.t_load)    == self.total_layers, "Mismatch: t_load vs total_layers"
        assert len(self.t_release) == self.total_layers, "Mismatch: t_release vs total_layers"

        # compute how many layers can stay resident under memory_limit
        self.layer_max = 0
        mem = 0
        for sz in self.layer_sizes:
            if mem + sz <= self.memory_limit:
                self.layer_max += 1
                mem += sz
            else:
                break

        print("layer_max:", self.layer_max)

        self.all_actions = [(k, w)
                            for k in range(self.layer_max + 1)
                            for w in range(1, self.layer_max - k + 1)]
        print("self.all_actions:", self.all_actions)

    def evaluate(self, k, w):
        # invalid combos get large negative reward
        if k + w > self.layer_max or w < 1 or k < 0:
            return -1e6

        D = self.total_layers - k
        start_time = sum(self.t_compute[:k])
        L = [0.0] * D
        C = [0.0] * D

        for j in range(D):
            idx = k + j
            if j < w:
                L[j] = (L[j-1] if j > 0 else 0) + self.t_load[idx]
            else:
                release_done = C[j-w] + self.t_release[idx - w]
                L[j] = max(L[j-1], release_done) + self.t_load[idx]

            compute_ready = start_time if j == 0 else C[j-1]
            C[j] = max(L[j], compute_ready) + self.t_compute[idx]

        return -C[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Bandit-based (k,w) optimization and shared memory writing."
    )
    parser.add_argument(
        "-m", "--memory_limit", type=float, required=True,
        help="Memory limit in MB"
    )
    parser.add_argument(
        "-s", "--shared_file_path", type=str, required=True,
        help="Path to shared memory file"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="Path to JSON config file"
    )
    args = parser.parse_args()

    env = BanditEnv(memory_limit=args.memory_limit, config_path=args.config)
    actions = env.all_actions
    n_actions = len(actions)

    action_counts = [0] * n_actions
    action_rewards = [0.0] * n_actions

    c = 1.0
    best_reward = -float('inf')
    best_action = None

    for episode in range(1, 501):
        ucb_scores = []
        for i in range(n_actions):
            if action_counts[i] == 0:
                ucb_scores.append(float('inf'))
            else:
                avg_r = action_rewards[i] / action_counts[i]
                bonus = c * math.sqrt(math.log(episode) / action_counts[i])
                ucb_scores.append(avg_r + bonus)

        action_idx = ucb_scores.index(max(ucb_scores))
        k, w = actions[action_idx]
        reward = env.evaluate(k, w)

        action_counts[action_idx] += 1
        action_rewards[action_idx] += reward

        if reward > best_reward:
            best_reward = reward
            best_action = (k, w)
        elif reward == best_reward:
            prev_k, prev_w = best_action
            if (k + w) > (prev_k + prev_w):
                best_action = (k, w)
            elif (k + w) == (prev_k + prev_w) and k > prev_k:
                best_action = (k, w)

        if episode % 50 == 0:
            print(
                f"[Episode {episode}] (k={k}, w={w}), "
                f"reward={reward:.2f}, best={best_action}, best_reward={best_reward:.2f}"
            )

    print(
        f"\n[Memory Limit = {args.memory_limit} MB] "
        f"best parameters: k={best_action[0]}, w={best_action[1]}, "
        f"inference time={-best_reward:.2f} ms"
    )

    # write best_action into shared memory
    k, w = best_action
    if not os.path.exists(args.shared_file_path):
        with open(args.shared_file_path, "wb") as f:
            f.truncate(8)

    with open(args.shared_file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 8)
        mm.seek(0)
        mm.write(struct.pack("ii", k, w))
        mm.flush()
        mm.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"memory-planning cost: {elapsed_time_ms:.2f} ms")
