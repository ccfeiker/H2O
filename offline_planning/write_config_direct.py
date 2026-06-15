import os
import mmap
import struct
import argparse

def write_k_w_to_shared_memory(k, w, shared_file_path):
    if not os.path.exists(shared_file_path):
        with open(shared_file_path, "wb") as f:
            f.truncate(8)

    with open(shared_file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 8)
        mm.seek(0)
        mm.write(struct.pack("ii", k, w))
        mm.flush()
        mm.close()
    print(f"Write to shared memory: k = {k}, w = {w} -> {shared_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Write (k, w) to shared memory file.")
    parser.add_argument("-k", type=int, required=True, help="Value of k")
    parser.add_argument("-w", type=int, required=True, help="Value of w")
    parser.add_argument("-s", "--shared_file_path", type=str, required=True, help="Path to shared memory file")
    args = parser.parse_args()

    write_k_w_to_shared_memory(args.k, args.w, args.shared_file_path)

if __name__ == "__main__":
    main()
