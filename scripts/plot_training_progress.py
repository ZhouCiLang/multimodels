#!/usr/bin/env python3
import re
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = Path('logs/train_unimodal_full.log')
text = log.read_text(encoding='utf-8', errors='ignore')
lines = text.splitlines()

# patterns
epoch_marker = re.compile(r"\b(\d{1,3})/50\b")
train_re = re.compile(r"train - Loss \(mean/total\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s+Avg\. embedding norm:\s*([0-9.]+)\s+Triplets per batch \(all/non-zero\):\s*([0-9.]+)\/([0-9.]+)")

entries = []
# walk lines, when we see a train_re, attempt to find nearest previous epoch marker
for idx, ln in enumerate(lines):
    m = train_re.search(ln)
    if m:
        # search backwards up to 20 lines for epoch marker
        epoch = None
        for back in range(1, 40):
            i = idx - back
            if i < 0: break
            mm = epoch_marker.search(lines[i])
            if mm:
                epoch = int(mm.group(1))
                break
        # fallback: if no epoch found, use None
        entries.append({
            'epoch': epoch,
            'mean_loss': float(m.group(1)),
            'total_loss': float(m.group(2)),
            'embed': float(m.group(3)),
            'trip_all': float(m.group(4)),
            'trip_nonzero': float(m.group(5)),
            'line_no': idx
        })

# If entries have None epoch, try to assign by proximity: use last seen epoch marker index
last_epoch_seen = None
epoch_positions = {}
for i, ln in enumerate(lines):
    mm = epoch_marker.search(ln)
    if mm:
        last_epoch_seen = (i, int(mm.group(1)))
        epoch_positions[i] = int(mm.group(1))

# For each entry with epoch None, find nearest epoch marker by line distance
for e in entries:
    if e['epoch'] is None:
        best = None
        bestd = 1e9
        for pos, ep in epoch_positions.items():
            d = abs(pos - e['line_no'])
            if d < bestd:
                bestd = d
                best = ep
        e['epoch'] = best

# sort entries by epoch then by line_no
entries.sort(key=lambda x: (x['epoch'] if x['epoch'] is not None else 9999, x['line_no']))

# aggregate per epoch: average values for entries with same epoch
from collections import defaultdict
agg = defaultdict(list)
for e in entries:
    if e['epoch'] is None: continue
    agg[e['epoch']].append(e)

rows = []
for ep in sorted(agg.keys()):
    vals = agg[ep]
    mean_loss = sum(v['mean_loss'] for v in vals)/len(vals)
    total_loss = sum(v['total_loss'] for v in vals)/len(vals)
    embed = sum(v['embed'] for v in vals)/len(vals)
    trip_nonzero = sum(v['trip_nonzero'] for v in vals)/len(vals)
    rows.append((ep, mean_loss, total_loss, embed, trip_nonzero, len(vals)))

# write CSV
out_csv = Path('logs/epoch_metrics.csv')
with out_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch','mean_loss','total_loss','embed_mean','trip_nonzero_mean','entries'])
    for r in rows:
        writer.writerow(r)

# make plots if we have points
if rows:
    epochs = [r[0] for r in rows]
    mean_losses = [r[1] for r in rows]
    embeds = [r[3] for r in rows]
    trip_nonzeros = [r[4] for r in rows]

    plt.figure(figsize=(6,4))
    plt.plot(epochs, mean_losses, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('mean_loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('logs/loss.png')

    plt.figure(figsize=(6,4))
    plt.plot(epochs, embeds, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('embed_norm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('logs/embed.png')

    plt.figure(figsize=(6,4))
    plt.plot(epochs, trip_nonzeros, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('trip_nonzero_mean')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('logs/triplets.png')

print('Wrote:', out_csv)
print('Saved plots: logs/loss.png, logs/embed.png, logs/triplets.png')
