#!/usr/bin/env python3
"""Auto-generate RESULTS.md from results/*.json + gender metrics from misclassified files."""

import os, json, glob, csv


def load_gender_map(data_dir):
    """Load (label, filename) -> gender mapping."""
    gender = {}
    meta = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(meta):
        return gender
    with open(meta) as f:
        for row in csv.DictReader(f):
            if row['split'] == 'test' and row.get('gender', '').strip() in ('M', 'F'):
                gender[(row['label'], os.path.basename(row['file_name']))] = row['gender'].strip()
    return gender


def compute_gender_acc(result_dir, gender_map):
    """Compute gender accuracy from misclassified files."""
    mis_dir = os.path.join(result_dir, 'misclassified_1x1')
    if not os.path.isdir(mis_dir):
        return None

    mis_set = set()
    for fn in os.listdir(mis_dir):
        if 'TRUE_' in fn and 'PRED_' in fn:
            parts = fn.split('__')
            true_cls = parts[0].replace('TRUE_', '')
            raw_fname = parts[2]
            short_fname = '_'.join(raw_fname.split('_')[1:])
            mis_set.add((true_cls, raw_fname))
            mis_set.add((true_cls, short_fname))

    m_tot = sum(1 for g in gender_map.values() if g == 'M')
    m_cor = sum(1 for (cls, fn), g in gender_map.items() if g == 'M' and (cls, fn) not in mis_set)
    f_tot = sum(1 for g in gender_map.values() if g == 'F')
    f_cor = sum(1 for (cls, fn), g in gender_map.items() if g == 'F' and (cls, fn) not in mis_set)

    if m_tot == 0 or f_tot == 0:
        return None

    m_acc = m_cor / m_tot
    f_acc = f_cor / f_tot
    return {'m_cor': m_cor, 'm_tot': m_tot, 'm_acc': m_acc,
            'f_cor': f_cor, 'f_tot': f_tot, 'f_acc': f_acc,
            'gap': f_acc - m_acc}


def find_data_dir():
    """Find the StimBench data directory."""
    import yaml
    for cfg in glob.glob('configs/*.yaml'):
        with open(cfg) as f:
            config = yaml.safe_load(f)
        data_dir = config.get('dataset', {}).get('path', '')
        if '/' in data_dir:
            # HuggingFace path — find local cache
            cache = os.path.expanduser('~/.cache/huggingface/hub')
            for root, dirs, files in os.walk(cache):
                if 'metadata.csv' in files and 'StimBench' in root:
                    return root
    return None


def main():
    results_files = sorted(glob.glob('results/*/results.json'))

    # Load gender map
    data_dir = find_data_dir()
    gender_map = load_gender_map(data_dir) if data_dir else {}
    has_gender = len(gender_map) > 0

    rows = []
    for path in results_files:
        with open(path) as f:
            data = json.load(f)
        name = data.get('experiment', os.path.basename(os.path.dirname(path)))
        result_dir = os.path.dirname(path)

        # Gender metrics
        g = compute_gender_acc(result_dir, gender_map) if has_gender else None

        for protocol, metrics in data.get('results', {}).items():
            row = {
                'model': name,
                'protocol': protocol,
                'acc': metrics['accuracy'],
                'f1w': metrics['f1_weighted'],
                'f1m': metrics['f1_macro'],
            }
            # Only add gender for 1x1 protocol
            if g and protocol == '1x1':
                row.update(g)
            rows.append(row)

    # Write RESULTS.md
    with open('RESULTS.md', 'w') as f:
        f.write('# StimBench Leaderboard\n\n')

        if has_gender:
            f.write(f'Gender test split: M={list(gender_map.values()).count("M")}, '
                    f'F={list(gender_map.values()).count("F")} stimming clips\n\n')
            f.write('| Model | Protocol | Acc | F1(w) | F1(m) | M Acc | F Acc | Gap (F-M) |\n')
            f.write('|-------|----------|-----|-------|-------|-------|-------|----------|\n')
        else:
            f.write('| Model | Protocol | Acc | F1(w) | F1(m) |\n')
            f.write('|-------|----------|-----|-------|-------|\n')

        for r in sorted(rows, key=lambda x: -x['f1w']):
            line = f"| {r['model']} | {r['protocol']} | {r['acc']:.4f} | {r['f1w']:.4f} | {r['f1m']:.4f}"
            if has_gender and 'm_acc' in r:
                gap_str = f"+{r['gap']:.1%}" if r['gap'] >= 0 else f"{r['gap']:.1%}"
                line += f" | {r['m_cor']}/{r['m_tot']} ({r['m_acc']:.1%}) | {r['f_cor']}/{r['f_tot']} ({r['f_acc']:.1%}) | {gap_str}"
            elif has_gender:
                line += " | — | — | —"
            line += " |"
            f.write(line + '\n')

    print(f"Generated RESULTS.md with {len(rows)} entries")
    if has_gender:
        print(f"  Gender metrics included (M={list(gender_map.values()).count('M')}, F={list(gender_map.values()).count('F')})")


if __name__ == '__main__':
    main()
