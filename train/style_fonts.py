"""Every family that trains the style encoder, in one shape: Google Fonts from bench/corpus.json and the open font files from
bench/open-fonts.json with their letters from the store's alphabets.json. An open family names its store, so `font_path`
finds its files, and its role comes from a fixed rule rather than a drawn split: a family whose id hashes to 0 in ten is
`held-out` for the open read, a role no training run names (the catalog runs take every Google role, `test` included),
whatever else is collected later; the rest `train`. The unit held out is a design, not a name: a family row that names its
`design` (a foundry's product page, under which Pangram Pangram sells a family's cuts and its variable twin) hashes that,
and a family's variable twin in the same source ("Sahel VF" beside "Sahel") hashes the family's id, so one design never
splits across the roles."""
import hashlib
import re

from scripts.corpus import ROOT, MANIFEST
from train.robustness import read

OPEN = ROOT/'bench/open-fonts.json'
HELD_OUT = 10  # one open family in ten is never trained on
LETTERS = 20   # a script counts when the family draws at least this many of its letters


def google_families():
    return [f for f in read(MANIFEST)['families'] if not f['excluded']]


def open_families(inventory=None, alphabets=None):
    """Open families with letters to draw, each carrying its store and alphabets as a Google family does."""
    inventory = inventory or read(OPEN); alphabets = alphabets or read(ROOT/inventory['store']/'alphabets.json')
    out = []
    for f in inventory['families']:
        letters = {script: a for script, a in alphabets.get(f['id'], {}).items() if len(a) >= LETTERS}
        if f.get('excluded') or f.get('symbolEncoded') or not letters: continue
        out.append({**f, 'store': inventory['store'], 'alphabets': letters})
    return out


def training_families():
    """Google's families first, then the open ones: faces enumerated from this keep every stored benchmark index."""
    return google_families() + open_families()


TWIN = re.compile(r'-(variable|vf)$')  # what a variable twin's id adds to its family's


def open_role(family_id, design=None):
    key = design or TWIN.sub('', family_id)
    return 'held-out' if int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % HELD_OUT == 0 else 'train'


def designs(families):
    """The design each family row names, by family id, for the role rule."""
    return {f['id']: f['design'] for f in families if f.get('design')}
