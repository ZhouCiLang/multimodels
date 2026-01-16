#!/usr/bin/env python3
import time
import re
import argparse
from pathlib import Path

def tail(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(0,2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line

def parse_percent(line):
    m = re.search(r"(\d+)%\|", line)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\[ *(\d+)/(\d+)\]", line)
    if m2:
        try:
            cur = int(m2.group(1)); tot = int(m2.group(2))
            return int(cur*100/tot)
        except Exception:
            return None
    return None

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--poll', type=float, default=1.0)
    args = p.parse_args()
    log = Path(args.log)
    print(f"Monitoring log: {log}")
    last_pct = -1
    if not log.exists():
        print("Log file does not exist yet, waiting...")
    for line in tail(args.log):
        pct = parse_percent(line)
        if pct is not None and pct != last_pct:
            bar = ('#' * (pct//2)).ljust(50)
            print(f"Progress: |{bar}| {pct}%")
            last_pct = pct
        # exit condition: detect final evaluation line
        if 'Final model results:' in line:
            print('Training finished (detected final results).')
            break
        time.sleep(args.poll)
