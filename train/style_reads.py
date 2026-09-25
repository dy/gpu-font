"""The shipped student read where the release read does not look: catalog validation queries rendered by WebKit, Firefox
and Chromium at 12 px (scripts/style-bench-engines.mjs packs), and the Chromium queries against the reference rows
rounded to 4 bits a number as shipped and product-quantized to 16 and 8 bytes a row. One reference embedding serves
every read. Writes bench/style-reads.json.

    python -m train.style_reads --run student-30k
"""
import argparse
import numpy as np
import torch

from scripts.corpus import ROOT
from train.encoder_data import save
from train.style import Setup, load_run, evaluate, references

KEEP = ('all', 'condition/', 'size/', 'script/Latn')


def product_quantized(vectors, subvectors, centroids=256, iterations=25, seed=0):
    """Rows rebuilt from per-subvector k-means codebooks trained on the rows themselves: what a row of `subvectors` bytes holds."""
    rng = np.random.default_rng(seed); parts = torch.from_numpy(vectors).float().chunk(subvectors, dim=1); out = []
    for part in parts:
        c = part[rng.choice(len(part), centroids, replace=False)].clone()
        for _ in range(iterations):
            code = torch.cdist(part, c).argmin(1)
            for k in range(centroids):
                members = part[code == k]
                if len(members): c[k] = members.mean(0)
        out.append(c[torch.cdist(part, c).argmin(1)])
    rows = torch.cat(out, 1).numpy(); return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def four_bit(vectors):
    """The shipped rounding: each row scaled to its largest magnitude at 7, rounded, normalized again."""
    q = np.round(vectors / np.abs(vectors).max(1, keepdims=True) * 7); return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def trim(report):
    """The groups worth keeping in the repository: whole, by condition, by size, Latin."""
    return {kind: {k: v for k, v in groups.items() if k.startswith(KEEP)} for kind, groups in report.items()}


def line(name, report):
    f = report['scriptFiltered']; g = lambda k: f"{100*f[k]['twin5']:.1f}" if k in f else ''
    print(f"{name:34} top5 {g('all')} first {100*f['all']['twin1']:.1f} clean {g('condition/clean')} degraded {g('condition/degraded')} 16 {g('size/16')} 24 {g('size/24')} 48 {g('size/48')} | unfiltered {100*report['unfiltered']['all']['twin5']:.1f}", flush=True)


def main(run):
    setup = Setup(['train']); model, heads, _ = load_run(run, exported=True)
    catalog = references(model, 'browser', 'mps'); torch.mps.empty_cache(); vectors, keys = catalog
    reads = {}
    def read(name, bench, rows=None):
        reads[name] = trim(evaluate(model, heads, setup, ['validation'], name=bench, source='browser', catalog=(rows, keys) if rows is not None else catalog)); line(name, reads[name]); torch.mps.empty_cache()
    for bench in ['catalog', 'catalog-webkit', 'catalog-firefox', 'catalog-chromium-12']: read(bench, bench)
    read('catalog-int4', 'catalog', four_bit(vectors))
    for m in [16, 8]: read(f'catalog-pq{m}', 'catalog', product_quantized(vectors, m))
    save(ROOT/'bench/style-reads.json', {'run': run, 'reports': reads, 'scope': 'The shipped student on catalog validation queries rendered by Chromium, WebKit, Firefox and Chromium at 12 px, against the shipped Chromium references; then the Chromium queries against the references rounded to 4 bits a number as shipped, and product-quantized to 16 and 8 bytes a row (k-means codebooks of 256 per subvector, trained on the references). Script-aware search, twin credit, as the main report.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', default='student-30k'); main(p.parse_args().run)
