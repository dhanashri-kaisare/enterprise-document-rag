import re
import subprocess
import sys

res = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_rag_eval.py", "-v", "--tb=short"],
    capture_output=True,
    text=True,
)
print(res.stdout)
print(res.stderr)

match = re.search(r"(\d+) passed", res.stdout)
passed = int(match.group(1)) if match else 0
pass_rate = (passed / 25) * 100
print(f"Evaluation Pass Rate: {passed}/25 ({pass_rate:.1f}%)")

sys.exit(0 if pass_rate >= 85.0 else 1)