"""Offline regression tests for the CPU-grouped FFT result aggregator."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import fft_v4_group_results as summary


class GroupedReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "downloaded"
        self.root.mkdir()
        self.output = Path(self.tmp.name) / "grouped"

    def add(self, slot, cpu="Example CPU 1", vendor="ExampleVendor", model="1",
            revision="pinned", factor=1.0, missing_case=False, invalid=False):
        folder = self.root / f"fft-v4-heuristic-experiments-{slot}"
        folder.mkdir()
        (folder / "REPORT.md").write_text("# Full benchmark report\n")
        (folder / "machine.json").write_text(json.dumps({
            "cpu_model": cpu, "vendor": vendor, "family": "7", "model": model,
            "stepping": "1", "microcode": "0x123",
            "compiler": "g++ synthetic", "cpu_id": 0,
        }))
        (folder / "commits.json").write_text(json.dumps([
            {"variant": x, "sha": revision, "target": "V2" if x == "V2" else "V4"}
            for x in summary.VARIANTS
        ]))
        fieldnames = ["group", *summary.CASE_FIELDS] + [
            f"{v}_us" for v in summary.VARIANTS
        ]
        with (folder / "all-cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for suite in summary.SUITES:
                for n in (1023,) if missing_case and suite == "extended" else (1023, 2047):
                    row = dict(group=suite, kind="fft", n=n, batch=16,
                               dtype="complex64", layout="C", pad=0, offset=0,
                               output="new")
                    for v, x in zip(summary.VARIANTS, (100, 80, 70, 60, 65, 55)):
                        row[f"{v}_us"] = factor * x
                    if invalid and suite == "legacy" and n == 1023:
                        row["D_us"] = 0
                    writer.writerow(row)
        return folder

    def test_all_runs_group_by_cpu_and_include_every_variant(self):
        self.add(1, factor=1.0)
        self.add(4, factor=1.3)
        self.add(7, cpu="Example CPU 2", model="2", factor=1.5)
        summary.summarize(self.root, self.output)
        report = (self.output / "CPU_GROUPED_REPORT.md").read_text()
        self.assertIn("3 / 16", report)
        self.assertIn("Example CPU 1 (2 runners)", report)
        self.assertIn("Example CPU 2 (1 runners)", report)
        self.assertIn("**Slots:** 1, 4", report)
        with (self.output / "CPU_GROUPED_RUNS.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3 * 2 * 6)
        d = [r for r in rows if r["slot"] == "4" and
             r["suite"] == "legacy" and r["variant"] == "D"][0]
        self.assertAlmostEqual(float(d["gm_vs_A"]), 80 / 65)
        with (self.output / "CPU_GROUPED_CASES.csv").open(newline="") as stream:
            cases = list(csv.DictReader(stream))
        self.assertEqual(len(cases), 2 * 2 * 2)
        c = next(x for x in cases if x["cpu_model"] == "Example CPU 1" and
                 x["suite"] == "legacy" and x["n"] == "1023")
        self.assertAlmostEqual(float(c["D_median_vs_A"]), 80 / 65)
        self.assertAlmostEqual(float(c["D_median_us"]), (65 + 65 * 1.3) / 2)

    def test_no_completed_run_fails_and_leaves_explanation(self):
        with self.assertRaisesRegex(RuntimeError, "No complete"):
            summary.summarize(self.root, self.output)
        self.assertIn("0 of 16", (self.output / "CPU_GROUPED_REPORT.md").read_text())

    def test_mismatched_manifest_is_not_silently_aggregated(self):
        self.add(1)
        self.add(2, revision="different")
        with self.assertRaisesRegex(ValueError, "commits differ"):
            summary.summarize(self.root, self.output)

    def test_missing_cases_fail(self):
        self.add(1)
        self.add(2, missing_case=True)
        with self.assertRaisesRegex(ValueError, "workload configurations differ"):
            summary.summarize(self.root, self.output)

    def test_bad_timings_fail(self):
        self.add(1, invalid=True)
        with self.assertRaisesRegex(ValueError, "invalid timing"):
            summary.summarize(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
