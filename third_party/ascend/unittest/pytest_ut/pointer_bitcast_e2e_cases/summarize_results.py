from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroupResult:
    name: str
    display_name: str
    expected: int
    exit_status: int
    xml_file: Path
    log_file: Path
    collected: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    duration: float = 0.0
    diagnostic: str = ""

    @property
    def state(self) -> str:
        if self.exit_status == 124:
            return "TIMEOUT"
        if self.diagnostic:
            return "NO XML"
        if self.collected != self.expected:
            return "COUNT MISMATCH"
        if self.exit_status != 0 or self.failed or self.errors:
            return "FAIL"
        if self.skipped or self.xfailed or self.xpassed:
            return "INCOMPLETE"
        return "PASS"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def count_log_outcome(log_text: str, label: str) -> int:
    matches = re.findall(rf"(?<!\w)(\d+)\s+{re.escape(label)}(?!\w)", log_text)
    return int(matches[-1]) if matches else 0


def parse_junit(result: GroupResult) -> None:
    if not result.xml_file.is_file():
        result.diagnostic = "JUnit XML was not produced"
        return

    try:
        root = ET.parse(result.xml_file).getroot()
    except (ET.ParseError, OSError) as exc:
        result.diagnostic = f"Cannot parse JUnit XML: {exc}"
        return

    test_cases = [node for node in root.iter() if local_name(node.tag) == "testcase"]
    result.collected = len(test_cases)

    for case in test_cases:
        result.duration += float(case.attrib.get("time", "0") or 0)
        children = {local_name(child.tag): child for child in case}
        if "error" in children:
            result.errors += 1
        elif "failure" in children:
            result.failed += 1
        elif "skipped" in children:
            skipped = children["skipped"]
            payload = " ".join(
                (
                    skipped.attrib.get("type", ""),
                    skipped.attrib.get("message", ""),
                    skipped.text or "",
                )
            ).lower()
            if "xfail" in payload:
                result.xfailed += 1
            else:
                result.skipped += 1
        else:
            result.passed += 1

    try:
        log_text = result.log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""

    reported_xpassed = count_log_outcome(log_text, "xpassed")
    if reported_xpassed:
        result.xpassed = reported_xpassed
        result.passed = max(0, result.passed - reported_xpassed)

    reported_xfailed = count_log_outcome(log_text, "xfailed")
    if reported_xfailed > result.xfailed:
        result.skipped = max(0, result.skipped - (reported_xfailed - result.xfailed))
        result.xfailed = reported_xfailed


def read_manifest(manifest: Path, details_dir: Path) -> list[GroupResult]:
    lines = manifest.read_text(encoding="utf-8").splitlines()
    results: list[GroupResult] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        name, display, expected, status, xml_name, log_name = line.split("\t")
        result = GroupResult(
            name=name,
            display_name=display,
            expected=int(expected),
            exit_status=int(status),
            xml_file=details_dir / xml_name,
            log_file=details_dir / log_name,
        )
        parse_junit(result)
        results.append(result)
    return results


def render_summary(args: argparse.Namespace, results: list[GroupResult]) -> tuple[str, bool]:
    expected = sum(result.expected for result in results)
    collected = sum(result.collected for result in results)
    passed = sum(result.passed for result in results)
    failed = sum(result.failed for result in results)
    errors = sum(result.errors for result in results)
    skipped = sum(result.skipped for result in results)
    xfailed = sum(result.xfailed for result in results)
    xpassed = sum(result.xpassed for result in results)
    duration = sum(result.duration for result in results)
    accepted = (
        len(results) == 4
        and expected == 46
        and collected == expected
        and passed == expected
        and failed == errors == skipped == xfailed == xpassed == 0
        and all(result.state == "PASS" for result in results)
    )

    lines = [
        "# Pointer Bitcast E2E Result",
        "",
        "## Overall",
        "",
        f"**{'PASS' if accepted else 'FAIL'}: {passed} passed, {failed} failed, "
        f"{errors} errors, {skipped} skipped, {xfailed} xfailed, "
        f"{xpassed} xpassed; {collected}/{expected} collected.**",
        "",
        f"- Commit: `{args.commit}`",
        f"- Started: `{args.started}`",
        f"- Finished: `{args.finished}`",
        f"- Device: `{args.device}`",
        f"- Overall timeout: `{args.timeout}s`",
        f"- Recorded test time: `{duration:.3f}s`",
        "",
        "## Groups",
        "",
        "| Group | State | Expected | Collected | Passed | Failed | Errors | Skipped | Xfailed | Xpassed | Exit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.display_name} | {result.state} | {result.expected} | "
            f"{result.collected} | {result.passed} | {result.failed} | "
            f"{result.errors} | {result.skipped} | {result.xfailed} | "
            f"{result.xpassed} | {result.exit_status} |"
        )

    diagnostics = [result for result in results if result.diagnostic]
    if diagnostics:
        lines.extend(("", "## Summary Diagnostics", ""))
        for result in diagnostics:
            lines.append(f"- `{result.name}`: {result.diagnostic}")

    lines.extend(
        (
            "",
            "## What To Return",
            "",
            "Send this file first. Do not send every detailed log initially.",
            "After the failing group is identified, use `test_results/README.md`",
            "to locate the requested log or XML slice.",
            "",
        )
    )
    return "\n".join(lines), accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--started", required=True)
    parser.add_argument("--finished", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--timeout", required=True)
    args = parser.parse_args()

    results = read_manifest(args.manifest, args.details_dir)
    summary, accepted = render_summary(args, results)
    args.output.write_text(summary, encoding="utf-8")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
