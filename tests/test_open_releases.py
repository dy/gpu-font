import json, tempfile, unittest
from pathlib import Path
from unittest import mock

from scripts import open_releases
from scripts.open_releases import api, choose, sh


class ReleaseFolder(unittest.TestCase):
    def test_the_folder_with_most_font_files_wins(self):
        paths = ['fonts/otf/A-Regular.otf', 'fonts/otf/A-Bold.otf', 'fonts/otf/A-Italic.otf', 'extra/A-Display.otf']
        self.assertEqual(choose(paths), ('fonts/otf', '.otf'))

    def test_opentype_wins_a_tie_with_truetype(self):
        paths = ['fonts/ttf/A-Regular.ttf', 'fonts/ttf/A-Bold.ttf', 'fonts/otf/A-Regular.otf', 'fonts/otf/A-Bold.otf']
        self.assertEqual(choose(paths), ('fonts/otf', '.otf'))

    def test_web_builds_sources_and_old_versions_are_never_the_release(self):
        paths = ['webfonts/A.ttf', 'webfonts/B.ttf', 'webfonts/C.ttf', 'sources/A.ttf', 'old/A.otf', 'old/B.otf', 'fonts/A.otf']
        self.assertEqual(choose(paths), ('fonts', '.otf'))

    def test_a_repository_with_only_sources_or_web_files_has_no_release(self):
        self.assertIsNone(choose(['sources/A.glyphs', 'web/A.woff2', 'web/A.ttf', 'README.md']))
        self.assertIsNone(choose([]))

    def test_files_at_the_root_are_a_release(self):
        self.assertEqual(choose(['A-Regular.otf', 'README.md']), ('.', '.otf'))


class CommandFailure(unittest.TestCase):
    """A failure is recorded by its first line, so that line must name the command and say what went wrong."""

    def test_the_command_s_own_message_is_kept(self):
        with self.assertRaises(RuntimeError) as failure: sh('git', 'ls-remote', '/nonexistent/repository')
        first = str(failure.exception).splitlines()[0]
        self.assertTrue(first.startswith('git ls-remote: '), first)
        self.assertIn('/nonexistent/repository', first)

    def test_a_failure_without_a_message_gives_the_exit_code(self):
        with self.assertRaises(RuntimeError) as failure: sh('false')
        self.assertEqual(str(failure.exception), 'false: exit 1')

    def test_output_is_returned(self):
        self.assertEqual(sh('echo', 'pinned'), 'pinned\n')


class GitHubQuota(unittest.TestCase):
    """Reproduces the run that recorded 2,788 repositories as failed when the hourly quota ran out."""
    SPENT = RuntimeError('gh api: gh: API rate limit exceeded for user ID 1. (HTTP 403)')
    SECONDARY = RuntimeError('gh api: gh: You have exceeded a secondary rate limit. (HTTP 403)')
    quota = staticmethod(lambda remaining, reset: json.dumps({'resources': {'core': {'remaining': remaining, 'reset': reset}}}))

    def call(self, *answers):
        with mock.patch.object(open_releases, 'sh', side_effect=answers) as run, \
             mock.patch.object(open_releases.time, 'sleep') as sleep, mock.patch.object(open_releases.time, 'time', return_value=1000):
            return api('repos/a/b/commits/main'), run, sleep

    def test_an_answer_needs_no_wait(self):
        result, run, sleep = self.call('{"sha": "c0ffee"}')
        self.assertEqual(result, {'sha': 'c0ffee'})
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    def test_a_spent_quota_is_waited_out_until_its_reset_then_retried(self):
        result, run, sleep = self.call(self.SPENT, self.quota(0, 1600), '{"sha": "c0ffee"}')
        self.assertEqual(result, {'sha': 'c0ffee'})
        sleep.assert_called_once_with(605)  # reset 1600 - now 1000, plus 5 seconds
        self.assertEqual([c.args for c in run.call_args_list], [('gh', 'api', 'repos/a/b/commits/main'), ('gh', 'api', 'rate_limit'), ('gh', 'api', 'repos/a/b/commits/main')])

    def test_a_reset_already_past_waits_only_the_margin(self):
        _, _, sleep = self.call(self.SPENT, self.quota(0, 900), '{}')
        sleep.assert_called_once_with(5)

    def test_a_secondary_limit_with_quota_left_waits_a_minute(self):
        _, _, sleep = self.call(self.SECONDARY, self.quota(4000, 1600), '{}')
        sleep.assert_called_once_with(60)

    def test_repeated_limits_keep_waiting(self):
        result, run, sleep = self.call(self.SPENT, self.quota(0, 1600), self.SECONDARY, self.quota(12, 1600), '[]')
        self.assertEqual(result, [])
        self.assertEqual([c.args for c in sleep.call_args_list], [(605,), (60,)])

    def test_any_other_failure_is_raised_at_once_with_gh_s_message(self):
        with self.assertRaises(RuntimeError) as failure: self.call(RuntimeError('gh api: gh: Not Found (HTTP 404)'))
        self.assertEqual(str(failure.exception), 'gh api: gh: Not Found (HTTP 404)')


class LargeTree(unittest.TestCase):
    """A tree past the API's listing limit, or too large to send, is listed by git instead of failing."""
    def pin(self, tree):
        answers = {'repos/a/b': {'default_branch': 'main', 'license': {'spdx_id': 'OFL-1.1'}},
                   'repos/a/b/commits/main': {'sha': 'api-commit', 'commit': {'tree': {'sha': 't'}}}}
        def api(path):
            if path in answers: return answers[path]
            if isinstance(tree, Exception): raise tree
            return tree
        with mock.patch.object(open_releases, 'api', side_effect=api), \
             mock.patch.object(open_releases, 'remote', return_value=('git-commit', ['fonts/A.otf'], 'https://github.com/a/b.git', None)) as git:
            return open_releases.github('a/b'), git

    def test_a_small_tree_is_listed_by_the_api(self):
        result, git = self.pin({'truncated': False, 'tree': [{'path': 'fonts/A.otf', 'type': 'blob'}, {'path': 'fonts', 'type': 'tree'}]})
        self.assertEqual(result, ('api-commit', ['fonts/A.otf'], 'https://github.com/a/b/archive/api-commit.zip', 'OFL-1.1'))
        git.assert_not_called()

    def test_a_truncated_or_unsendable_tree_is_listed_by_git_at_git_s_commit(self):
        for tree in [{'truncated': True, 'tree': []}, RuntimeError('gh api: unexpected end of JSON input')]:
            result, git = self.pin(tree)
            self.assertEqual(result, ('git-commit', ['fonts/A.otf'], 'https://github.com/a/b/archive/git-commit.zip', 'OFL-1.1'))
            git.assert_called_once_with('github.com', 'a/b')


class SharedRepository(unittest.TestCase):
    def test_a_link_into_a_folder_takes_that_family_not_the_largest(self):
        paths = ['Big/A.otf', 'Big/B.otf', 'Big/C.otf', 'MetaAccanthis/MetaAccanthis.otf', 'LICENSE']
        with mock.patch.object(open_releases, 'remote', return_value=('c0ffee', paths, 'https://gitlab.com/a/fonts.git', None)) as git:
            row = open_releases.pin({'source': 'use-and-modify', 'family': 'MetaAccanthis', 'repo': 'https://gitlab.com/a/fonts/-/tree/master/MetaAccanthis'})
            whole = open_releases.pin({'source': 'use-and-modify', 'family': 'Big', 'repo': 'https://gitlab.com/a/fonts'})
        self.assertEqual(git.call_args_list[0].args, ('gitlab.com', 'a/fonts'))
        self.assertEqual(row['include'], r'^[^/]+/MetaAccanthis/[^/]+\.otf$')
        self.assertEqual(whole['include'], r'^[^/]+/Big/[^/]+\.otf$')

    def test_a_github_folder_link_and_a_branch_link_without_a_folder(self):
        paths = ['Big/A.ttf', 'Big/B.ttf', 'FairfaxHD/FairfaxHD.ttf', 'FairfaxHD/OFL.txt']
        with mock.patch.object(open_releases, 'github', return_value=('c0ffee', paths, 'https://github.com/k/relay/archive/c0ffee.zip', None)) as api:
            folder = open_releases.pin({'source': 'uncut', 'family': 'Fairfax HD', 'repo': 'https://github.com/k/relay/tree/master/FairfaxHD/'})
            branch = open_releases.pin({'source': 'uncut', 'family': 'Big', 'repo': 'https://github.com/k/relay/tree/master'})
        self.assertEqual(api.call_args_list[0].args[0], 'k/relay')
        self.assertEqual(folder['include'], r'^[^/]+/FairfaxHD/[^/]+\.ttf$')
        self.assertEqual(branch['include'], r'^[^/]+/Big/[^/]+\.ttf$')  # a branch alone names no folder: the whole repository


class Run(unittest.TestCase):
    def test_a_run_records_the_first_line_of_a_failure_and_a_second_run_changes_nothing(self):
        rows = [{'source': 'github', 'family': 'Pinned', 'repo': 'https://github.com/a/pinned'},
                {'source': 'github', 'family': 'Gone', 'repo': 'https://github.com/a/gone'},
                {'source': 'velvetyne', 'family': 'No repository'}]

        def pin(row):
            if row['family'] == 'Gone': raise RuntimeError('git ls-remote: remote: Repository not found.\nfatal: repository not found')
            return {**row, 'commit': 'c0ffee'}

        with tempfile.TemporaryDirectory() as folder, mock.patch('sys.stdout'), mock.patch('sys.stderr'):
            listing = Path(folder) / 'open-releases.json'
            listing.write_text(json.dumps({'releases': rows}))
            with mock.patch.object(open_releases, 'LISTING', listing), mock.patch.object(open_releases, 'pin', side_effect=pin) as pinned:
                open_releases.main()
                first = listing.read_text()
                open_releases.main()
            saved = json.loads(first)['releases']
            self.assertEqual([row.get('commit') for row in saved], ['c0ffee', None, None])
            self.assertEqual(saved[1]['failed'], 'git ls-remote: remote: Repository not found.')
            self.assertEqual(pinned.call_count, 2)  # the second run retries neither the pinned nor the failed row
            self.assertEqual(listing.read_text(), first)
            self.assertEqual([path.name for path in Path(folder).iterdir()], ['open-releases.json'])  # nothing half-written left


if __name__ == '__main__':
    unittest.main()
