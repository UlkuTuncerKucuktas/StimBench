#!/usr/bin/env python3
"""Build RESULTS.md from results/*/results.json: one table per training regime ranked by macro F1,
per-class recall, gender split, and a cross-regime comparison. `--tex` also writes results/leaderboard.tex."""

import os, sys, json, glob, csv

REGIMES = [('_realsyn', 'real + synthetic'), ('_synonly', 'synthetic only'), ('', 'real')]
SHORT = {'armflapping': 'AF', 'headbanging': 'HB', 'normal': 'N', 'spinning': 'Sp'}


def split_regime(name):
    for suffix, label in REGIMES:
        if suffix and name.endswith(suffix):
            return name[:-len(suffix)], suffix
    return name, ''


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def gender_split(rows):
    out = {}
    for g in ('M', 'F'):
        sub = [r for r in rows if r.get('gender', '').strip() == g]
        if sub:
            out[g] = (sum(int(r['correct']) for r in sub), len(sub))
    return out if len(out) == 2 else None


def gender_from_misclassified(result_dir, gender_map):
    mis_dir = os.path.join(result_dir, 'misclassified_1x1')
    if not os.path.isdir(mis_dir) or not gender_map:
        return None
    wrong = set()
    for fn in os.listdir(mis_dir):
        if 'TRUE_' in fn and 'PRED_' in fn:
            parts = fn.split('__')
            cls, raw = parts[0].replace('TRUE_', ''), parts[2]
            wrong.add((cls, raw))
            wrong.add((cls, '_'.join(raw.split('_')[1:])))
    out = {}
    for g in ('M', 'F'):
        keys = [k for k, v in gender_map.items() if v == g]
        if keys:
            out[g] = (sum(k not in wrong for k in keys), len(keys))
    return out if len(out) == 2 else None


def load_gender_map():
    cache = os.path.expanduser('~/.cache/huggingface/hub')
    for root, dirs, files in os.walk(cache):
        if 'metadata.csv' in files and 'StimBench' in root:
            rows = read_csv(os.path.join(root, 'metadata.csv'))
            return {(r['label'], os.path.basename(r['file_name'])): r['gender'].strip()
                    for r in rows if r['split'] == 'test' and r.get('gender', '').strip() in ('M', 'F')}
    return {}


def load_result(path, gender_map):
    with open(path) as f:
        data = json.load(f)
    result_dir = os.path.dirname(path)
    name = data.get('experiment', os.path.basename(result_dir))
    base, suffix = split_regime(name)
    classes = data['config']['dataset']['classes']
    protocols = data.get('results', {})
    main = protocols.get('1x1') or next(iter(protocols.values()))
    cm = main['confusion_matrix']
    recall = {c: (row[i], sum(row)) for i, (c, row) in enumerate(zip(classes, cm))}
    preds = read_csv(os.path.join(result_dir, 'predictions_1x1.csv'))
    gender = gender_split(preds) or gender_from_misclassified(result_dir, gender_map)
    return {
        'name': name, 'base': base, 'suffix': suffix, 'classes': classes,
        'acc': main['accuracy'], 'f1w': main['f1_weighted'], 'f1m': main['f1_macro'],
        'recall': recall, 'gender': gender,
        'other': {p: m['accuracy'] for p, m in protocols.items() if p != '1x1'},
    }


def pct(cor, tot):
    return f'{cor}/{tot} ({cor / tot:.0%})' if tot else '—'


def regime_table(rows, classes):
    head = ['Model', 'Acc', 'F1(w)', 'F1(m)'] + [SHORT.get(c, c) for c in classes] + ['M acc', 'F acc', 'Gap (F−M)']
    others = sorted({p for r in rows for p in r['other']})
    head += [f'Acc {p}' for p in others]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in sorted(rows, key=lambda x: (-x['f1m'], -x['acc'])):
        cells = [r['base'], f"{r['acc']:.4f}", f"{r['f1w']:.4f}", f"{r['f1m']:.4f}"]
        cells += [pct(*r['recall'][c]) for c in classes]
        if r['gender']:
            (mc, mt), (fc, ft) = r['gender']['M'], r['gender']['F']
            cells += [pct(mc, mt), pct(fc, ft), f'{fc / ft - mc / mt:+.1%}']
        else:
            cells += ['—', '—', '—']
        cells += [f"{r['other'][p]:.4f}" if p in r['other'] else '—' for p in others]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def comparison_table(results, classes):
    by_base = {}
    for r in results:
        by_base.setdefault(r['base'], {})[r['suffix']] = r
    multi = {b: d for b, d in by_base.items() if len(d) > 1}
    if not multi:
        return ''
    suffixes = [s for s, _ in REGIMES if any(s in d for d in multi.values())]
    labels = dict(REGIMES)
    head = ['Model'] + [f'F1(m) {labels[s]}' for s in suffixes] + [f'AF recall {labels[s]}' for s in suffixes]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    af = 'armflapping' if 'armflapping' in classes else classes[0]
    for base, d in sorted(multi.items(), key=lambda kv: -max(r['f1m'] for r in kv[1].values())):
        cells = [base]
        cells += [f"{d[s]['f1m']:.4f}" if s in d else '—' for s in suffixes]
        cells += [pct(*d[s]['recall'][af]) if s in d else '—' for s in suffixes]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def tex_table(results, classes):
    labels = dict(REGIMES)
    lines = [r'\begin{tabular}{ll' + 'r' * (3 + len(classes)) + '}', r'\toprule',
             'Model & Regime & Acc & F1(w) & F1(m) & ' + ' & '.join(SHORT.get(c, c) for c in classes) + r' \\', r'\midrule']
    for r in sorted(results, key=lambda x: (x['base'], [s for s, _ in REGIMES].index(x['suffix']))):
        rec = ' & '.join(f"{c}/{t}" for c, t in (r['recall'][k] for k in classes))
        model = r['base'].replace('_', '\\_')
        lines.append(f"{model} & {labels[r['suffix']]} & {r['acc']:.4f} & {r['f1w']:.4f} & {r['f1m']:.4f} & {rec} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def main():
    gender_map = load_gender_map()
    results = [load_result(p, gender_map) for p in sorted(glob.glob('results/*/results.json'))]
    if not results:
        print('No results found')
        return
    classes = results[0]['classes']
    out = ['# StimBench Leaderboard', '', 'Ranked by macro F1 on the 1×1 protocol; per-class cells are recall (correct/total).', '']
    for suffix, label in REGIMES:
        rows = [r for r in results if r['suffix'] == suffix]
        if rows:
            out += [f'## Trained on {label}', '', regime_table(rows, classes), '']
    comparison = comparison_table(results, classes)
    if comparison:
        out += ['## Across training regimes', '', comparison, '']
    with open('RESULTS.md', 'w') as f:
        f.write('\n'.join(out))
    print(f'Generated RESULTS.md with {len(results)} runs')
    if '--tex' in sys.argv:
        with open('results/leaderboard.tex', 'w') as f:
            f.write(tex_table(results, classes) + '\n')
        print('Wrote results/leaderboard.tex')


if __name__ == '__main__':
    main()
