"""Pin catalogue families to their upstream repositories, for scripts/open_fonts.py.

bench/open-releases.json lists families found in open catalogues (Velvetyne,
Collletttivo, Uncut and so on), each with its licence, the page stating it, and
its upstream repository. This fills in what the inventory needs to fetch it
reproducibly: the current commit, its archive URL, and which font files to take.
Rows already pinned are left alone; delete a row's `commit` to re-pin it.

    python scripts/open_releases.py        pin unpinned rows (GitHub via gh, other hosts via git)
"""
import json, re, subprocess, sys, tempfile, urllib.parse
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LISTING = ROOT / 'bench/open-releases.json'
# Folders that hold web builds, old versions or working files rather than the release.
SKIP = re.compile(r'(^|/)(web|webfonts?|woff2?|eot|svg|old|archive|previous|test|tests|proof|proofs|specimens?|documentation|docs|sources?|src|ufo|glyphs|build-tools?|scripts)(/|$)', re.I)


def github(repo):
    run = lambda *path: json.loads(subprocess.run(['gh', 'api', '/'.join(path)], check=True, capture_output=True, text=True).stdout)
    meta = run('repos', repo)
    commit = run('repos', repo, 'commits', meta['default_branch'])['sha']
    tree = run('repos', repo, f'git/trees/{commit}?recursive=1')
    if tree.get('truncated'): raise ValueError(f'{repo}: tree too large to list')
    licence = (meta.get('license') or {}).get('spdx_id')  # GitHub's reading of the repository's own licence file
    return commit, [item['path'] for item in tree['tree'] if item['type'] == 'blob'], f'https://github.com/{repo}/archive/{commit}.zip', licence


def remote(host, repo):
    """Over git itself, for GitLab, Codeberg or any other host: GitLab challenges plain archive
    downloads and keeps bots off its API, and git is what every host serves to programs."""
    url = f'https://{host}/{repo}.git'
    git = lambda *args, **kw: subprocess.run(['git', *args], check=True, capture_output=True, text=True, **kw).stdout
    commit = git('ls-remote', url, 'HEAD').split()[0]
    with tempfile.TemporaryDirectory() as folder:
        git('clone', '--quiet', '--filter=blob:none', '--no-checkout', url, folder)
        paths = git('-C', folder, 'ls-tree', '-r', '--name-only', commit).splitlines()
    return commit, paths, url, None


def choose(paths):
    """The release folder: the folder and format holding the most font files outside web builds
    and sources. OpenType wins a tie; a repository with only web or source files yields None."""
    groups = defaultdict(list)
    for path in paths:
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in ('.otf', '.ttf') and not SKIP.search(str(PurePosixPath(path).parent)): groups[(str(PurePosixPath(path).parent), suffix)].append(path)
    if not groups: return None
    (folder, suffix), _ = max(groups.items(), key=lambda pair: (len(pair[1]), pair[0][1] == '.otf', -len(pair[0][0])))
    return folder, suffix


def pin(row):
    link = urllib.parse.urlparse(row['repo'])
    parts = [part for part in link.path.split('/-/')[0].split('/') if part]
    # GitHub repositories are owner/name, whatever the link points to inside; GitLab allows groups.
    repo = '/'.join(parts[:2] if link.netloc == 'github.com' else parts).removesuffix('.git')
    if len(parts) < 2: raise ValueError(f'not a repository link: {row["repo"]}')
    commit, paths, url, licence = github(repo) if link.netloc == 'github.com' else remote(link.netloc, repo)
    row = {**row, 'commit': commit, **({'repoLicence': licence} if licence else {})}
    choice = choose(paths)
    if not choice: return {**row, 'unusable': 'no OpenType or TrueType files outside web builds and sources'}
    folder, suffix = choice
    # Archives unpack into "<name>-<commit>/"; the pattern matches below that prefix.
    inside = '' if folder == '.' else re.escape(folder) + '/'
    return {**row, 'url': url, 'include': rf'^[^/]+/{inside}[^/]+\{suffix}$', **({'format': 'git'} if url.endswith('.git') else {})}


def main():
    listing = json.loads(LISTING.read_text())
    for index, row in enumerate(listing['releases']):
        if row.get('commit') or not row.get('repo'): continue
        try: listing['releases'][index] = pin(row)
        except Exception as error: print(f"{row['family']}: {error}", file=sys.stderr); continue
        done = listing['releases'][index]
        print(f"{row['source']:14} {row['family']:30} {done.get('include') or done.get('unusable')}")
    LISTING.write_text(json.dumps(listing, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
