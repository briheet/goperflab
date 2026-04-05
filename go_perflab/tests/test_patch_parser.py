import tempfile
import unittest
from pathlib import Path

from models import GoPerfAction
from server.env import GoPerfEnvironment


class PatchParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.env = GoPerfEnvironment(workspace_root=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _repo_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "go-bench-repo"

    def _init_repo(self) -> None:
        self.env.reset(task_id="easy")
        init_action = GoPerfAction(
            action_type="init_repo",
            repo_path=str(self._repo_path()),
            repo_copy=True,
            repo_init_if_missing=True,
        )
        obs = self.env.step(init_action)
        self.assertEqual(obs.exit_code, 0, msg=obs.stderr)

    def _assert_patch_applies(self, patch_diff: str) -> None:
        action = GoPerfAction(action_type="patch", patch_diff=patch_diff)
        obs = self.env.step(action)
        self.assertEqual(obs.exit_code, 0, msg=obs.stderr)
        workspace = Path(self.env.state.workspace_path or "")
        contents = (workspace / "string_concatenation" / "string.go").read_text()
        self.assertIn("strings.Builder", contents)

    def test_unified_patch_applies(self) -> None:
        self._init_repo()
        patch = """diff --git a/string_concatenation/string.go b/string_concatenation/string.go
--- a/string_concatenation/string.go
+++ b/string_concatenation/string.go
@@ -6,10 +6,10 @@
 
 func calculateResult(num int) string {
 
-\tvar result string
-\tfor i := range num {
-\t\tresult += fmt.Sprintf("item-%d,", i)
+\tvar result strings.Builder
+\tfor i := 0; i < num; i++ {
+\t\tresult.WriteString(fmt.Sprintf("item-%d,", i))
 \t}
 
-\treturn result
+\treturn result.String()
 }
"""
        self._assert_patch_applies(patch)

    def test_unified_patch_with_extra_space_before_tabs(self) -> None:
        self._init_repo()
        patch = """diff --git a/string_concatenation/string.go b/string_concatenation/string.go
--- a/string_concatenation/string.go
+++ b/string_concatenation/string.go
@@ -6,10 +6,10 @@
 
func calculateResult(num int) string {
 
- \tvar result string
- \tfor i := range num {
- \t\tresult += fmt.Sprintf("item-%d,", i)
+ \tvar result strings.Builder
+ \tfor i := 0; i < num; i++ {
+ \t\tresult.WriteString(fmt.Sprintf("item-%d,", i))
 \t}
 
- \treturn result
+ \treturn result.String()
 }
"""
        self._assert_patch_applies(patch)

    def test_unified_patch_missing_context_prefix(self) -> None:
        self._init_repo()
        patch = """diff --git a/string_concatenation/string.go b/string_concatenation/string.go
--- a/string_concatenation/string.go
+++ b/string_concatenation/string.go
@@ -6,10 +6,10 @@

func calculateResult(num int) string {

-\tvar result string
-\tfor i := range num {
-\t\tresult += fmt.Sprintf("item-%d,", i)
+\tvar result strings.Builder
+\tfor i := 0; i < num; i++ {
+\t\tresult.WriteString(fmt.Sprintf("item-%d,", i))
\t}

-\treturn result
+\treturn result.String()
}
"""
        self._assert_patch_applies(patch)



if __name__ == "__main__":
    unittest.main()
