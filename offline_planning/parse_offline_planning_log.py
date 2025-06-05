#!/usr/bin/env python3

import re
import json
import argparse
import os
import time


def parse_log(log_path):
    io_speeds     = []
    layer_sizes   = []
    compute_times = {}
    release_times = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1) IO Speed
            m = re.search(r"IO Speed:(\d+)\s*MB/s", line)
            if m:
                io_speeds.append(int(m.group(1)))

            # 2) Layer size
            m = re.search(r"layer size:(\d+)\s*MB", line)
            if m:
                layer_sizes.append(int(m.group(1)))

            # 3) Compute time for numeric blk indices
            m = re.search(r"compute index:blk\.(\d+)\s*compute time:(\d+)\s*ms", line)
            if m:
                idx = int(m.group(1))
                compute_times[idx] = int(m.group(2))

            # 4) Release time (note the comma)
            m = re.search(r"release index:blk\.(\d+),\s*release time:(\d+)\s*ms", line)
            if m:
                idx = int(m.group(1))
                release_times[idx] = int(m.group(2))

    # Calculate average IO Speed and round
    io_speed = round(sum(io_speeds) / len(io_speeds)) if io_speeds else 0

    # Sort by index and extract time lists
    t_compute = [compute_times[i] for i in sorted(compute_times)]
    t_release = [release_times[i] for i in sorted(release_times)]

    return {
        "io_speed": io_speed,
        "layer_sizes":   layer_sizes,
        "t_compute":     t_compute,
        "t_release":     t_release,
    }

def write_single_line_json(data, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as fw:
        fw.write('{\n')
        fw.write(f'  "io_speed": {data["io_speed"]},\n')
        fw.write(f'  "layer_sizes": {json.dumps(data["layer_sizes"], ensure_ascii=False)},\n')
        fw.write(f'  "t_compute": {json.dumps(data["t_compute"],   ensure_ascii=False)},\n')
        fw.write(f'  "t_release": {json.dumps(data["t_release"],   ensure_ascii=False)}\n')
        fw.write('}\n')

def main():
    parser = argparse.ArgumentParser(
        description="Parse offline_planning_log and generate a JSON configuration file."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/tmp/offline_planning_log",
        help="Path to the input log file (default: /tmp/offline_planning_log)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./model_offline_config/model_config.json",
        help="Path to the output JSON file (default: ./model_offline_config/model_config.json)"
    )
    args = parser.parse_args()

    data = parse_log(args.log_path)
    write_single_line_json(data, args.output_path)
    print(f"Generated JSON file at: {args.output_path}")

if __name__ == "__main__":
    main()
