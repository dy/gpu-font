"""Openly licensed font repositories on GitHub, listed for scripts/open_releases.py.

Many designers publish open fonts only in their own repositories, in no catalogue.
GitHub's search finds them by licence: every OFL repository, and repositories
tagged font, fonts or typeface under another open licence. Each new repository
becomes a row in bench/open-releases.json (source "github", `raw`), which
open_releases.py pins to a commit and to the font files of its release folder;
open_fonts.py then takes each family unless a source earlier in its order has it.

    python scripts/github_fonts.py        add repositories not yet listed
"""
import datetime, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTING = ROOT / 'bench/open-releases.json'
OPEN = ['mit', 'apache-2.0', 'cc0-1.0', 'unlicense', 'cc-by-4.0', 'cc-by-sa-4.0', 'gpl-2.0', 'gpl-3.0', 'lgpl-3.0', 'mpl-2.0', 'bsd-3-clause', 'wtfpl']
QUERIES = ['license:ofl-1.1', *(f'topic:{topic} license:{licence}' for topic in ('font', 'fonts', 'typeface') for licence in OPEN)]
FIRST = datetime.date(2008, 1, 1)  # GitHub opened in 2008
CAP = 1000  # search returns at most this many results per query


def search(query, page=1):
    time.sleep(2.1)  # the search API allows 30 requests a minute
    result = subprocess.run(['gh', 'api', '-X', 'GET', 'search/repositories', '-f', f'q={query}', '-f', 'per_page=100', '-f', f'page={page}'],
                            capture_output=True, text=True)
    if result.returncode:  # a secondary rate limit: wait it out once
        time.sleep(60)
        result = subprocess.run(['gh', 'api', '-X', 'GET', 'search/repositories', '-f', f'q={query}', '-f', 'per_page=100', '-f', f'page={page}'],
                                capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def repositories(query, start=FIRST, end=None):
    """Every repository the query finds, splitting the creation dates until each part fits the cap."""
    end = end or datetime.date.today()
    scoped = f'{query} fork:false created:{start}..{end}'
    first = search(scoped)
    if first['total_count'] > CAP and start < end:
        middle = start + (end - start) / 2
        yield from repositories(query, start, middle)
        yield from repositories(query, middle + datetime.timedelta(days=1), end)
        return
    yield from first['items']
    for page in range(2, min(first['total_count'], CAP) // 100 + 2):
        yield from search(scoped, page)['items']


def main():
    listing = json.loads(LISTING.read_text())
    known = {row['repo'].rstrip('/').lower() for row in listing['releases'] if row.get('repo')}
    added = 0
    for query in QUERIES:
        for repo in repositories(query):
            url = repo['html_url'].rstrip('/').lower()
            if url in known or repo.get('fork'): continue
            known.add(url)
            listing['releases'].append({'source': 'github', 'family': repo['name'], 'page': repo['html_url'],
                                        'licence': (repo.get('license') or {}).get('spdx_id') or 'see repository',
                                        'repo': repo['html_url'], 'branch': repo['default_branch'], 'raw': True})
            added += 1
        LISTING.write_text(json.dumps(listing, ensure_ascii=False, indent=1) + '\n')
        print(f'{query}: {added} repositories listed so far', flush=True)


if __name__ == '__main__':
    main()
