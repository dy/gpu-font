"""Text lines in the DaFont forum's request pictures, boxed by macOS Vision as scripts/text_boxes.py boxes the Commons
photographs: the first picture of every identified request (the one the asker posted), each line with its pixel box,
the words read and Vision's confidence, largest first. Appends to .data/forum/dafont/boxes.jsonl, so a stopped run goes
on where it left off; a reader and a packer take the lines from there.

    python scripts/forum_boxes.py          # macOS only (pyobjc-framework-Vision)
"""
import json
import sys
from pathlib import Path

from scripts.text_boxes import lines_of

ROOT = Path(__file__).resolve().parents[1]
FORUM = ROOT/'.data/forum/dafont'


def main():
    out = FORUM/'boxes.jsonl'
    done = {json.loads(line)['file'] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    requests = [json.loads(line) for line in (FORUM/'requests.jsonl').read_text().splitlines() if line.strip()]
    todo = [(r, r['files'][0]['file']) for r in requests if r.get('identified') and r.get('files') and r['files'][0]['file'] not in done]
    print(f'{len(todo)} identified requests to box, {len(done)} done', flush=True)
    with out.open('a') as stream:
        for n, (r, file) in enumerate(todo):
            try: found = lines_of(FORUM/'images'/file)
            except Exception as error: found = None; print(f'{file}: {error}', file=sys.stderr)
            stream.write(json.dumps({'file': file, 'id': r['id'], **(found or {'lines': None})}) + '\n'); stream.flush()
            if (n + 1) % 500 == 0: print(f'{n + 1}/{len(todo)}', flush=True)


if __name__ == '__main__': main()
