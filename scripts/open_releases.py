"""Pin catalogue families to their upstream repositories, for scripts/open_fonts.py.

bench/open-releases.json lists families found in open catalogues (Velvetyne,
Collletttivo, Uncut and so on), each with its licence, the page stating it, and
its upstream repository. This fills in what the inventory needs to fetch it
reproducibly: the current commit, its archive URL, and which font files to take.
Rows already pinned are left alone; delete a row's `commit` to re-pin it.

    python scripts/open_releases.py        pin unpinned rows (GitHub via gh, other hosts via git)
"""
import json, re, subprocess, sys, tempfile, time, urllib.parse
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_FONTS = 80  # a release folder rarely holds more; beyond that it is a build dump
LICENCE = re.compile(r'(LICEN[CS]E|COPYING|OFL)[^/]*$', re.I)
LISTING = ROOT / 'bench/open-releases.json'
# Folders that hold web builds, old versions or working files rather than the release.
SKIP = re.compile(r'(^|/)(web|webfonts?|woff2?|eot|svg|old|archive|previous|test|tests|proof|proofs|specimens?|documentation|docs|sources?|src|ufo|glyphs|build-tools?|scripts)(/|$)', re.I)


def sh(*command):
    """A command's output. A failure carries the command's own message, and its first line is what gets recorded."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode: raise RuntimeError(f"{' '.join(command[:2])}: {result.stderr.strip() or f'exit {result.returncode}'}")
    return result.stdout


def api(path):
    """GitHub's REST API through gh, waiting out the hourly quota rather than failing on it."""
    while True:
        try: return json.loads(sh('gh', 'api', path))
        except RuntimeError as error:
            if 'rate limit' not in str(error).lower(): raise
        core = json.loads(sh('gh', 'api', 'rate_limit'))['resources']['core']  # costs no quota
        # The hourly quota spent: wait for its reset. Otherwise a secondary limit, which lifts within a minute.
        time.sleep(max(core['reset'] - time.time(), 0) + 5 if not core['remaining'] else 60)


def github(repo, branch=None):
    run = lambda *path: api('/'.join(path))
    meta = {} if branch else run('repos', repo)  # a search result already names the branch
    head = run('repos', repo, 'commits', branch or meta['default_branch'])
    commit = head['sha']
    licence = (meta.get('license') or {}).get('spdx_id')  # GitHub's reading of the repository's own licence file
    try: tree = run('repos', repo, f"git/trees/{head['commit']['tree']['sha']}?recursive=1")
    except RuntimeError: tree = {'truncated': True}  # a tree too large even to send whole
    if tree.get('truncated'):  # past the API's listing limit: git lists a tree of any size
        commit, paths, _, _ = remote('github.com', repo)
        return commit, paths, f'https://github.com/{repo}/archive/{commit}.zip', licence
    return commit, [item['path'] for item in tree['tree'] if item['type'] == 'blob'], f'https://github.com/{repo}/archive/{commit}.zip', licence


def remote(host, repo):
    """Over git itself, for GitLab, Codeberg or any other host: GitLab challenges plain archive
    downloads and keeps bots off its API, and git is what every host serves to programs."""
    url = f'https://{host}/{repo}.git'
    git = lambda *args: sh('git', *args)
    commit = git('ls-remote', url, 'HEAD').split()[0]
    with tempfile.TemporaryDirectory() as folder:
        git('clone', '--quiet', '--depth', '1', '--filter=blob:none', '--no-checkout', url, folder)  # the tip's paths, no history, no file bodies
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
    commit, paths, url, licence = github(repo, row.get('branch')) if link.netloc == 'github.com' else remote(link.netloc, repo)
    # A link into a folder (GitHub's …/tree/main/Family, GitLab's …/-/tree/main/Family) names one family of a shared repository.
    inside = re.search(r'/tree/[^/]+/(.+?)/?$', link.path)
    if inside: paths = [path for path in paths if path.startswith(urllib.parse.unquote(inside.group(1)) + '/')]  # fonts/Old%20Round is fonts/Old Round
    row = {**row, 'commit': commit, **({'repoLicence': licence} if licence else {})}
    choice = choose(paths)
    if not choice: return {**row, 'unusable': 'no OpenType or TrueType files outside web builds and sources'}
    folder, suffix = choice
    if row.get('raw') and link.netloc == 'github.com':
        # Just the release folder's fonts and the root licence, as files at the pinned commit:
        # a long-tail repository's archive is mostly sources.
        fonts = [path for path in paths if str(PurePosixPath(path).parent) == folder and path.lower().endswith(suffix)][:MAX_FONTS]
        licences = [path for path in paths if '/' not in path and LICENCE.match(path)]
        raw = lambda path: f'https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}'
        return {**row, 'files': [raw(path) for path in fonts + licences], 'format': 'file', 'folder': repo.replace('/', '--'), 'include': rf'\{suffix}$'}
    # Archives unpack into "<name>-<commit>/"; the pattern matches below that prefix.
    inside = '' if folder == '.' else re.escape(folder) + '/'
    return {**row, 'url': url, 'include': rf'^[^/]+/{inside}[^/]+\{suffix}$', **({'format': 'git'} if url.endswith('.git') else {})}


def main():
    listing = json.loads(LISTING.read_text())
    def save():  # written aside and renamed, so an interrupted run never leaves half a file
        LISTING.with_suffix('.tmp').write_text(json.dumps(listing, ensure_ascii=False, indent=1) + '\n')
        LISTING.with_suffix('.tmp').replace(LISTING)
    for index, row in enumerate(listing['releases']):
        if row.get('commit') or row.get('failed') or not row.get('repo'): continue
        try: listing['releases'][index] = pin(row)
        except Exception as error:
            # Recorded so a long run does not retry it; delete `failed` to try again.
            listing['releases'][index] = {**row, 'failed': str(error).splitlines()[0][:200]}
            print(f"{row['family']}: {error}", file=sys.stderr); continue
        done = listing['releases'][index]
        print(f"{row['source']:14} {row['family']:30} {done.get('include') or done.get('unusable')}", flush=True)
        if index % 50 == 0: save()  # a run of thousands survives interruption
    save()


if __name__ == '__main__':
    main()
