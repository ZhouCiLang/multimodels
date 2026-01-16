#!/usr/bin/env python3
import re
from pathlib import Path
import statistics

logpath = Path('logs/train_unimodal_full.log')
text = logpath.read_text(encoding='utf-8', errors='ignore')
lines = text.splitlines()

# find epoch markers like '  2%|▏         | 1/50 '
epoch_re = re.compile(r"\b(\d{1,3})/50\b")
train_re = re.compile(r"train - Loss \(mean/total\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s+Avg\. embedding norm:\s*([0-9.]+)\s+Triplets per batch \(all/non-zero\):\s*([0-9.]+)\/([0-9.]+)")

# map epoch_index (1-based) -> list of stats
epoch_stats = {i: [] for i in range(1,51)}
current_epoch = None

for ln in lines:
    m = epoch_re.search(ln)
    if m:
        # epoch marker found; use the captured number as epoch index
        idx = int(m.group(1))
        # some markers are repeated for same epoch; set current epoch
        current_epoch = idx
        continue
    if current_epoch is None:
        continue
    m2 = train_re.search(ln)
    if m2:
        mean_loss = float(m2.group(1))
        total_loss = float(m2.group(2))
        embed = float(m2.group(3))
        trip_all = float(m2.group(4))
        trip_nonzero = float(m2.group(5))
        epoch_stats[current_epoch].append((mean_loss, total_loss, embed, trip_all, trip_nonzero))

# summarize per epoch
summary = []
for e in range(1,51):
    vals = epoch_stats[e]
    if not vals:
        summary.append((e, None))
        continue
    mean_losses = [v[0] for v in vals]
    total_losses = [v[1] for v in vals]
    embeds = [v[2] for v in vals]
    trip_all = [v[3] for v in vals]
    trip_nonzero = [v[4] for v in vals]
    summary.append((e, {
        'mean_loss_mean': statistics.mean(mean_losses),
        'mean_loss_std': statistics.pstdev(mean_losses) if len(mean_losses)>1 else 0.0,
        'total_loss_mean': statistics.mean(total_losses),
        'embed_mean': statistics.mean(embeds),
        'trip_all_mean': statistics.mean(trip_all),
        'trip_nonzero_mean': statistics.mean(trip_nonzero),
        'samples': len(vals)
    }))

# extract final evaluation block
eval_results = {}
cur_dataset = None
for i, ln in enumerate(lines):
    if ln.startswith('Final model results:'):
        # parse following lines
        for j in range(i+1, i+200):
            if j>=len(lines): break
            l = lines[j].strip()
            if l.startswith('Dataset:'):
                parts = l.split()
                if len(parts)>=2:
                    cur_dataset = parts[1]
            elif l.startswith('Avg. top 1% recall:') and cur_dataset:
                m = re.search(r"Avg\. top 1% recall:\s*([0-9.]+)\s*Avg\. top 1 recall:\s*([0-9.]+)", l)
                if m:
                    eval_results[cur_dataset] = {
                        'top1_percent': float(m.group(1)),
                        'top1': float(m.group(2))
                    }
            elif l=='' and cur_dataset:
                cur_dataset = None

# print table
print('Epoch | mean_loss_mean | total_loss_mean | embed_mean | trip_nonzero_mean | samples')
print('-----|---------------:|----------------:|-----------:|------------------:|--------')
for e, stat in summary:
    if stat is None:
        print(f'{e:3d}   |       N/A      |       N/A       |     N/A    |        N/A        |   0')
    else:
        print(f"{e:3d}   | {stat['mean_loss_mean']:<14.4f} | {stat['total_loss_mean']:<14.4f} | {stat['embed_mean']:<10.4f} | {stat['trip_nonzero_mean']:<16.2f} | {stat['samples']}")

print('\nFinal evaluation results:')
for ds, vals in eval_results.items():
    print(f"- {ds}: Avg top1% = {vals['top1_percent']}, Avg top1 = {vals['top1']}")
