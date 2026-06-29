"""propose: structured check results and pass/fail reporting for pipeline gates.
input: named check results with pass/fail status.
output: console summary and boolean gate status."""

from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    required: bool = True


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.passed else ("WARN" if not result.required else "FAIL")
    print(f"[{status}] {result.name}: {result.detail}")


def summarize(results: list[CheckResult], title: str) -> bool:
    print(f"\n=== {title} ===")
    for result in results:
        print_result(result)
    required_failures = [r for r in results if r.required and not r.passed]
    print(f"\nRequired checks: {len(results) - len([r for r in results if not r.required])}")
    print(f"Required failures: {len(required_failures)}")
    if required_failures:
        print("\nGate status: BLOCKED")
        return False
    print("\nGate status: PASSED")
    return True
