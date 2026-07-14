import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Sample 3 rows from each pipe-count group to map all schemas
with open('ml_training/data/combined_2026.csv', 'r', encoding='utf-8', errors='replace') as f:
    header = f.readline()
    lines = f.readlines()

print(f"Header: {header.strip()}")
print()

from collections import defaultdict
groups = defaultdict(list)
for line in lines:
    pc = line.count('|')
    if len(groups[pc]) < 3:
        groups[pc].append(line.strip())

for pc in sorted(groups):
    print(f"=== {pc} pipes ({pc+1} cols) ===")
    for row in groups[pc]:
        parts = row.split('|')
        print(f"  [{len(parts)} fields]")
        for i, p in enumerate(parts):
            print(f"    [{i:02d}] {p[:60]}")
    print()
