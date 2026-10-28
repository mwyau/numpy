"""Run and report isolated DUCC FFT CPU-dispatch ASV profiles in CI."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("FFT_BENCH_ROOT", Path(__file__).resolve().parents[2])).resolve()
BENCH_FILTER = r"^bench_fft\."
TARGETS = ("X86_V2", "X86_V3", "X86_V4")
PROFILES = TARGETS + ("POCKETFFT", "LEGACY_V3")
# Non-quick ASV: 14 samples over two rounds, with per-case warmup.
TIMING_ATTRIBUTES = {"warmup_time": 0.15, "repeat": 7, "rounds": 2}
PROFILE_MASKS = {
    "X86_V2": "X86_V4,X86_V3",
    "X86_V3": "X86_V4",
    "LEGACY_V3": "X86_V4",
    "X86_V4": "",
}
RELEVANT_FLAGS = {
    "sse3", "pni", "ssse3", "sse4_1", "sse4_2", "popcnt", "cx16",
    "lahf_lm", "avx", "avx2", "f16c", "fma", "bmi1", "bmi2",
    "lzcnt", "abm", "movbe", "avx512f", "avx512cd", "avx512bw",
    "avx512dq", "avx512vl",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_path(name: str) -> Path:
    artifact_root = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    destination = artifact_root / f"fft-dispatch-{name.lower()}"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cpu_flags() -> list[str]:
    flags: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"flags", "features"}:
                flags.update(value.split())
                break
    except OSError:
        pass
    return sorted(flags & RELEVANT_FLAGS)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"}:
                return value.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def import_built_numpy():
    import numpy as np
    from numpy._core._multiarray_umath import __cpu_features__
    from numpy.fft import _duccfft_umath

    for label, module_file in (
        ("NumPy", np.__file__),
        ("DUCC FFT extension", _duccfft_umath.__file__),
    ):
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(ROOT):
            raise RuntimeError(
                f"{label} loaded outside the checked-out source build: {resolved}"
            )

    return np, __cpu_features__, _duccfft_umath



def import_built_pocketfft():
    """Ensure the reference uses the actual pinned main-branch PocketFFT build."""
    import importlib.util

    import numpy as np
    from numpy.fft import _pocketfft_umath

    expected = os.environ.get("FFT_POCKETFFT_COMMIT")
    actual = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if not expected or expected != actual:
        raise RuntimeError(f"PocketFFT main commit mismatch: {actual} != {expected}")
    if os.environ.get("NPY_DISABLE_CPU_FEATURES"):
        raise RuntimeError("Unexpected CPU feature mask on main-branch PocketFFT.")
    if importlib.util.find_spec("numpy.fft._duccfft_umath") is not None:
        raise RuntimeError("PocketFFT reference unexpectedly includes a DUCC extension.")
    for label, filename in (
        ("NumPy", np.__file__),
        ("PocketFFT extension", _pocketfft_umath.__file__),
    ):
        path = Path(filename).resolve()
        if not path.is_relative_to(ROOT / "build-install"):
            raise RuntimeError(f"{label} imported outside PocketFFT build: {path}")
    print(f"PocketFFT NumPy main commit: {actual}")
    print(f"PocketFFT NumPy version: {np.__version__}")
    print(f"PocketFFT extension: {_pocketfft_umath.__file__}")
    return np, {
        "selected_target": "PocketFFT (main)",
        "compiled_targets": [],
        "reference_commit": actual,
        "numpy_version": np.__version__,
        "numpy_file": str(Path(np.__file__).resolve()),
        "extension_file": str(Path(_pocketfft_umath.__file__).resolve()),
    }


def dispatch_report(features, extension) -> dict:
    return {
        "compiled_targets": extension.__duccfft_cpu_dispatch_targets__.split(),
        "selected_target": extension.__duccfft_cpu_dispatch_target__,
        "cpu_feature_groups": {
            target: bool(features.get(target, False)) for target in TARGETS
        },
    }


def show_numpy_build(np, features, extension) -> dict:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
        if hasattr(np, "show_runtime"):
            np.show_runtime()
    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"Architecture: {platform.machine()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Relevant CPU flags: {' '.join(cpu_flags())}")
    print(f"NumPy: {np.__version__} ({Path(np.__file__).resolve()})")
    print(f"DUCC FFT extension: {Path(extension.__file__).resolve()}")
    print(f"CPU feature groups: {json.dumps({target: bool(features.get(target, False)) for target in TARGETS}, sort_keys=True)}")
    report = dispatch_report(features, extension)
    print(f"Compiled DUCC FFT targets: {' '.join(report['compiled_targets'])}")
    print(f"Selected DUCC FFT target: {report['selected_target']}")
    print("NumPy build configuration and runtime:")
    print(output.getvalue(), end="")
    return report


def host_profile() -> int:
    output = artifact_path("diagnostics")
    data = {
        "status": "failed",
        "started_at": now_utc(),
        "architecture": platform.machine(),
        "cpu_model": cpu_model(),
        "relevant_cpu_flags": cpu_flags(),
    }
    try:
        np, features, extension = import_built_numpy()
        report = show_numpy_build(np, features, extension)
        data.update(report)
        data.update({
            "status": "passed",
            "numpy_version": np.__version__,
            "numpy_file": str(Path(np.__file__).resolve()),
            "extension_file": str(Path(extension.__file__).resolve()),
            "python": sys.version,
            "compiler": subprocess.run(
                ["c++", "--version"], capture_output=True, text=True,
                check=False,
            ).stdout.splitlines()[0] if shutil_which("c++") else "unavailable",
        })
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        write_json(output / "host.json", data)
        return 1
    finally:
        data["finished_at"] = now_utc()
    write_json(output / "host.json", data)
    return 0


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)


def verify_fft_correctness(np) -> None:
    source = np.array([1.0, 2.0, 3.0, 4.0])
    spectrum = np.array([10.0, -2.0 + 2.0j, -2.0, -2.0 - 2.0j])
    np.testing.assert_allclose(np.fft.fft(source), spectrum)
    np.testing.assert_allclose(np.fft.ifft(spectrum), source)
    np.testing.assert_allclose(np.fft.irfft(np.fft.rfft(source), n=4), source)

    for dtype in (np.complex64, np.complex128):
        values = (np.arange(8, dtype=np.float32)
                  + 1j * np.arange(8, dtype=np.float32)[::-1]).astype(dtype)
        recovered = np.fft.ifft(np.fft.fft(values))
        np.testing.assert_allclose(recovered, values, rtol=1e-5, atol=1e-5)

    for dtype in (np.float32, np.float64):
        tolerance = 1e-5 if dtype is np.float32 else 1e-12
        for size in (7, 8):
            values = np.arange(size, dtype=dtype)
            recovered = np.fft.irfft(np.fft.rfft(values), n=size)
            np.testing.assert_allclose(
                recovered, values, rtol=tolerance, atol=tolerance
            )


def baseline_is_v2(target: str) -> bool:
    match = re.fullmatch(r"baseline\(([^)]+)\)", target)
    return bool(match and match.group(1) == "X86_V2")


def assert_profile(profile: str, host: dict, features, report: dict) -> str:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError(
            f"Host architecture is {platform.machine()}, expected x86-64."
        )

    expected_mask = PROFILE_MASKS[profile]
    actual_mask = os.environ.get("NPY_DISABLE_CPU_FEATURES", "")
    if actual_mask != expected_mask:
        raise RuntimeError(
            f"Incorrect feature mask for {profile}: expected {expected_mask!r}, "
            f"received {actual_mask!r}."
        )

    if host.get("status") != "passed":
        raise RuntimeError(
            "Unmasked CPU/build diagnostic failed before profile execution: "
            + str(host.get("error", "host report is missing or incomplete"))
        )

    if not host.get("cpu_feature_groups", {}).get("X86_V2", False):
        raise RuntimeError("Runner does not expose the required x86-64-v2 CPU features.")

    compiled = report["compiled_targets"]
    selected = report["selected_target"]

    if profile in ("X86_V3", "LEGACY_V3"):
        if not host.get("cpu_feature_groups", {}).get("X86_V3", False):
            raise RuntimeError(
                "Runner does not expose X86_V3 CPU features; the V3 benchmark "
                "profile cannot run on this host."
            )
        if "X86_V3" not in compiled:
            raise RuntimeError(
                "Compiler/build missing the X86_V3 DUCC FFT dispatch target."
            )
        if not features.get("X86_V3", False):
            raise RuntimeError(
                "X86_V3 was available before masking, but NumPy does not report "
                "it as executable in the masked process."
            )
        if selected != "X86_V3":
            raise RuntimeError(
                "Runtime dispatch selected a lower target unexpectedly for X86_V3: "
                f"{selected}."
            )
        return "X86_V3"

    if profile == "X86_V4":
        if not host.get("cpu_feature_groups", {}).get("X86_V4", False):
            raise RuntimeError(
                "Runner does not expose the required x86-64-v4 CPU features; "
                "AVX-512 benchmark validation could not run."
            )
        if "X86_V4" not in compiled:
            raise RuntimeError(
                "Compiler/build missing the X86_V4 DUCC FFT dispatch target."
            )
        if not features.get("X86_V4", False):
            raise RuntimeError(
                "The unmasked host reported X86_V4, but the profile process does "
                "not report it as executable."
            )
        if selected != "X86_V4":
            raise RuntimeError(
                "Runtime dispatch selected a lower target unexpectedly for X86_V4: "
                f"{selected}."
            )
        return "X86_V4"

    if profile == "X86_V2":
        if "X86_V2" not in compiled and not any(
            baseline_is_v2(target) for target in compiled
        ):
            raise RuntimeError(
                "Compiler/build missing both the X86_V2 DUCC FFT target and "
                "an X86_V2 baseline."
            )
        if features.get("X86_V3", False) or features.get("X86_V4", False):
            raise RuntimeError(
                "The V2 profile feature mask did not disable X86_V3 and X86_V4."
            )
        if selected != "X86_V2" and not baseline_is_v2(selected):
            raise RuntimeError(
                "Runtime dispatch selected a lower target unexpectedly for X86_V2: "
                f"{selected}; expected X86_V2 or baseline(X86_V2)."
            )
        return selected

    raise ValueError(f"Unknown CPU profile: {profile}")


def count_cases(benchmarks) -> int:
    count = 0
    for name in benchmarks:
        selected = benchmarks.benchmark_selection[name]
        if selected is None:
            count += 1
        else:
            count += len(selected)
    return count


def record_asv_results(benchmarks, results) -> tuple[list[dict], dict]:
    rows = []
    totals = {"completed": 0, "failed": 0, "skipped": 0}
    for name, benchmark in benchmarks.items():
        selected = benchmarks.benchmark_selection.get(name)
        selected_indices = (
            list(selected) if selected is not None
            else [0]
        )
        values = results._results.get(name)
        raw_samples = results._samples.get(name) or []
        error_code = results.errcode.get(name, 0)
        params = benchmark.get("params") or []

        for index in selected_indices:
            if values is None or index >= len(values):
                value = None
            else:
                value = values[index]
            case_params = (
                list(next(itertools.islice(itertools.product(*params), index, index + 1)))
                if params else []
            )

            if value is None:
                status = "failed" if error_code else "skipped"
                serialized_value = None
            elif isinstance(value, (int, float)) and math.isnan(float(value)):
                status = "skipped"
                serialized_value = None
            else:
                status = "completed"
                serialized_value = value

            totals[status] += 1
            rows.append({
                "benchmark": name,
                "parameters": case_params,
                "status": status,
                "asv_value": serialized_value,
                "samples": raw_samples[index] if index < len(raw_samples) else None,
            })

    return rows, totals


def run_profile(profile: str) -> int:
    directory = artifact_path(profile)
    result_file = directory / "results.json"
    data = {
        "profile": profile,
        "expected_target": profile,
        "feature_mask": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
        "started_at": now_utc(),
        "status": "failed",
        "collected_cases": 0,
        "completed_cases": 0,
        "failed_cases": 0,
        "skipped_cases": 0,
        "asv_values_file": "asv-results.json",
        "timing_attributes": TIMING_ATTRIBUTES,
    }
    started = time.monotonic()
    return_code = 1
    try:
        if profile == "POCKETFFT":
            np, report = import_built_pocketfft()
            data.update(report)
        else:
            host_file = artifact_path("diagnostics") / "host.json"
            host = json.loads(host_file.read_text())
            np, features, extension = import_built_numpy()
            report = show_numpy_build(np, features, extension)
            data.update(report)
            data["numpy_version"] = np.__version__
            data["numpy_file"] = str(Path(np.__file__).resolve())
            data["extension_file"] = str(Path(extension.__file__).resolve())
            data["host_feature_groups"] = host.get("cpu_feature_groups", {})
            data["selected_target"] = assert_profile(profile, host, features, report)

        verify_fft_correctness(np)
        print(f"{profile}: representative FFT correctness checks passed.")

        os.chdir(ROOT / "benchmarks")
        from asv.commands import run as asv_run
        from asv.config import Config

        conf = Config.load(str(ROOT / "benchmarks" / "asv.conf.json"))
        captured = []
        native_run_benchmarks = asv_run.run_benchmarks
        print(f"Running ASV with calibrated, warmed, repeated samples: {TIMING_ATTRIBUTES}")

        def capture_results(benchmarks, environment, **kwargs):
            discovered = count_cases(benchmarks)
            data["collected_cases"] = discovered
            print(f"ASV discovered {discovered} FFT parameter cases.")
            if discovered != 190:
                raise RuntimeError(
                    f"ASV discovered {discovered} cases, expected all 190."
                )
            benchmark_started = time.monotonic()
            result = native_run_benchmarks(benchmarks, environment, **kwargs)
            data["benchmark_execution_seconds"] = round(
                time.monotonic() - benchmark_started, 3
            )
            captured.append((benchmarks, result))
            return result

        asv_run.run_benchmarks = capture_results
        try:
            machine = os.environ.get("ASV_MACHINE_NAME", socket.gethostname())
            asv_status = asv_run.Run.run(
                conf,
                bench=[BENCH_FILTER],
                quick=False,
                attribute=TIMING_ATTRIBUTES,
                record_samples=True,
                show_stderr=True,
                env_spec=["existing:same"],
                pull=False,
                machine=machine,
            )
            return_code = 0 if asv_status is None else asv_status
        finally:
            asv_run.run_benchmarks = native_run_benchmarks

        if captured:
            benchmarks, results = captured[-1]
            rows, totals = record_asv_results(benchmarks, results)
            write_json(directory / "asv-results.json", {
                "profile": profile,
                "selected_target": data.get("selected_target"),
                "reference_commit": data.get("reference_commit"),
                "cases": rows,
            })
            data.update({
                "completed_cases": totals["completed"],
                "failed_cases": totals["failed"],
                "skipped_cases": totals["skipped"],
            })
        else:
            data["error"] = "ASV did not invoke its benchmark runner."

        if return_code != 0:
            data["error"] = f"ASV exited with status {return_code}."
            data["failure_category"] = "benchmark-execution"
        elif not captured:
            return_code = 1
        elif data["completed_cases"] != 190 or data["failed_cases"] or data["skipped_cases"]:
            data["error"] = (
                "ASV case totals did not show 190 completed cases with no "
                "failures or skips."
            )
            data["failure_category"] = "benchmark-execution"
            return_code = 1
        else:
            data["status"] = "passed"
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
        data["failure_category"] = classify_error(profile, str(exc))
        traceback.print_exc()
        return_code = 1
    finally:
        data["duration_seconds"] = round(time.monotonic() - started, 3)
        data["finished_at"] = now_utc()
        data["exit_code"] = return_code
        write_json(result_file, data)
        print(f"{profile} summary: {json.dumps(data, sort_keys=True)}")
    return return_code


def classify_error(profile: str, message: str) -> str:
    if "x86-64-v4 CPU features" in message:
        return "host-cpu-missing-v4"
    if "X86_V3 CPU features" in message:
        return "host-cpu-missing-v3"
    if "Compiler/build missing" in message:
        return "compiler-or-build-target-missing"
    if "Runtime dispatch selected" in message:
        return "runtime-dispatch-selected-lower-target"
    if "ASV" in message or "case" in message.lower():
        return "benchmark-discovery-or-execution"
    if profile == "X86_V4":
        return "v4-preflight-or-benchmark"
    return "preflight-or-benchmark"



def render_performance_report() -> tuple[str, str, dict]:
    """Compare identical ASV cases from one runner. Times are seconds."""
    by_mode: dict[str, dict[tuple[str, tuple[str, ...]], dict]] = {}
    for mode in TARGETS:
        path = artifact_path(mode) / "asv-results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        by_mode[mode] = {
            (row["benchmark"], tuple(row["parameters"])): row
            for row in payload["cases"]
        }

    if not by_mode:
        return (
            "## Performance comparison\n\nNo completed ASV profiles available.\n",
            "benchmark\tparameters\tv2_us\tv3_us\tv4_us\tv3_over_v2\tv4_over_v2\n",
            {"common_cases": 0},
        )

    keys = sorted(set().union(*(set(rows) for rows in by_mode.values())))
    totals = {"X86_V3": [], "X86_V4": []}
    grouped: dict[str, dict[str, list[float]]] = {}
    slowdowns: list[tuple[float, str, str, str]] = []
    lines = ["benchmark\tparameters\tv2_us\tv3_us\tv4_us\tv3_over_v2\tv4_over_v2"]

    def valid(row: dict | None) -> float | None:
        if row is None or row.get("status") != "completed":
            return None
        value = row.get("asv_value")
        if isinstance(value, (float, int)) and math.isfinite(value) and value > 0:
            return float(value)
        return None

    for key in keys:
        bench_name, params = key
        values = {mode: valid(rows.get(key)) for mode, rows in by_mode.items()}
        v2 = values.get("X86_V2")
        ratios = {}
        for mode in ("X86_V3", "X86_V4"):
            other = values.get(mode)
            ratio = v2 / other if v2 is not None and other is not None else None
            ratios[mode] = ratio
            if ratio is not None:
                totals[mode].append(ratio)
                category = bench_name.split(".")[1] if "." in bench_name else bench_name
                grouped.setdefault(category, {}).setdefault(mode, []).append(ratio)
                slowdowns.append((ratio, mode, bench_name, ", ".join(params)))
        cells = [bench_name, ", ".join(params)]
        cells.extend(f"{values[mode]*1e6:.6f}" if values.get(mode) is not None else ""
                     for mode in TARGETS)
        cells.extend(f"{ratios[mode]:.6f}" if ratios.get(mode) is not None else ""
                     for mode in ("X86_V3", "X86_V4"))
        lines.append("\t".join(cells))

    def geometric(values: list[float]) -> float | None:
        return math.exp(statistics.fmean(math.log(x) for x in values)) if values else None

    def fmt(ratio: float | None) -> str:
        return f"{ratio:.3f}×" if ratio is not None else "—"

    markdown = [
        "## Warmed, repeated ASV performance comparison",
        "",
        "All ratios are **V2 time / other-target time**, so values above 1.00× favor the higher target.",
        "Times in the TSV artifact are microseconds. Comparisons use common completed cases only.",
        "This is a shared GitHub runner; CPU contention and frequency changes may affect results.",
        "",
        "### Overall",
        "",
        "| Target | Matched cases | Geometric-mean speedup vs V2 | Median speedup | Faster cases |",
        "|---|---:|---:|---:|---:|",
    ]
    summary = {}
    for mode in ("X86_V3", "X86_V4"):
        ratios = totals[mode]
        geo = geometric(ratios)
        summary[mode] = {
            "matched_cases": len(ratios),
            "geometric_mean_speedup": geo,
            "median_speedup": statistics.median(ratios) if ratios else None,
            "faster_cases": sum(r > 1 for r in ratios),
        }
        markdown.append(
            f"| {mode} | {len(ratios)} | {fmt(geo)} | "
            f"{fmt(statistics.median(ratios) if ratios else None)} | "
            f"{summary[mode]['faster_cases']}/{len(ratios)} |"
        )
    markdown += [
        "",
        "### By benchmark class (geometric mean)",
        "",
        "| Benchmark class | Cases compared V3 / V4 | V3/V2 | V4/V2 |",
        "|---|---:|---:|---:|",
    ]
    for name, modes in sorted(grouped.items()):
        v3, v4 = modes.get("X86_V3", []), modes.get("X86_V4", [])
        markdown.append(
            f"| {name} | {len(v3)} / {len(v4)} | "
            f"{fmt(geometric(v3))} | {fmt(geometric(v4))} |"
        )
    markdown += [
        "",
        "### Largest observed higher-target slowdowns",
        "",
        "| Profile | Benchmark | Parameters | V2/target |",
        "|---|---|---|---:|",
    ]
    for ratio, mode, bench, params in sorted(slowdowns)[:12]:
        markdown.append(f"| {mode} | {bench} | {params.replace('|','/')} | {ratio:.3f}× |")
    markdown += [
        "",
        "**Measurement method:** ASV non-quick, 0.15 s warmup, 7 repeats and 2 rounds; "
        "ASV controls calibration and returns medians. The same NumPy binary is used in "
        "separate processes with native NumPy CPU-dispatch feature masks; "
        "preflight checks assert each selected DUCC target. Each benchmark's setup "
        "is excluded from the timed function. All profiles are CPU-affinity-pinned "
        "to the same permitted logical CPU.",
        "",
        "See `fft-dispatch-timings.tsv` for every case and "
        "`asv-results.json` in each profile artifact for underlying ASV samples.",
        "",
    ]
    return "\n".join(markdown), "\n".join(lines) + "\n", summary



def render_pocketfft_comparison() -> tuple[str, str, dict]:
    """Compare pinned NumPy-main PocketFFT with the DUCC CPU variants."""
    root = artifact_path("POCKETFFT")
    header = ("benchmark\tparameters\tpocketfft_us\tv2_us\tv3_us\tv4_us\t"
              "pocketfft_over_v2\tpocketfft_over_v3\tpocketfft_over_v4")
    path = root / "asv-results.json"
    if not path.is_file():
        return ("## DUCC versus PocketFFT\n\nPocketFFT result unavailable.\n",
                header + "\n", {"status": "missing"})
    reference = json.loads(path.read_text())
    ref_rows = {(r["benchmark"], tuple(r["parameters"])): r
                for r in reference["cases"]}
    ducc_rows = {}
    for mode in TARGETS:
        other_path = artifact_path(mode) / "asv-results.json"
        if other_path.is_file():
            data = json.loads(other_path.read_text())
            ducc_rows[mode] = {(r["benchmark"], tuple(r["parameters"])): r
                               for r in data["cases"]}

    def measured(row):
        if not row or row.get("status") != "completed":
            return None
        number = row.get("asv_value")
        if not isinstance(number, (int, float)):
            return None
        number = float(number)
        return number if math.isfinite(number) and number > 0 else None

    ratios = {mode: [] for mode in TARGETS}
    grouped = {}
    table = [header]
    for key in sorted(ref_rows):
        name, params = key
        base = measured(ref_rows[key])
        timings = {mode: measured(ducc_rows.get(mode, {}).get(key))
                   for mode in TARGETS}
        row = [name, ", ".join(params),
               f"{base * 1e6:.6f}" if base is not None else ""]
        row += [f"{timings[mode] * 1e6:.6f}" if timings[mode] is not None else ""
                for mode in TARGETS]
        for mode in TARGETS:
            other = timings[mode]
            speedup = base / other if base is not None and other is not None else None
            row.append(f"{speedup:.6f}" if speedup is not None else "")
            if speedup is not None:
                ratios[mode].append(speedup)
                group = name.split(".")[1] if "." in name else name
                grouped.setdefault(group, {}).setdefault(mode, []).append(speedup)
        table.append("\t".join(row))

    def geo(numbers):
        return math.exp(statistics.fmean(math.log(x) for x in numbers)) if numbers else None

    def show(value):
        return f"{value:.3f}×" if value is not None else "—"

    ref_sha = reference.get("reference_commit", "unknown")
    report = [
        "## DUCC versus PocketFFT from NumPy main",
        "",
        f"PocketFFT reference commit: {ref_sha}. Both builds use the same "
        "190-case benchmark file and pinned CPU on the same GitHub runner.",
        "**Speedup = PocketFFT seconds / DUCC seconds** (>1 means DUCC faster).",
        "",
        "| DUCC mode | Cases | Geometric mean | Median | DUCC faster |",
        "|---|---:|---:|---:|---:|",
    ]
    summary = {"pocketfft_commit": ref_sha}
    for mode in TARGETS:
        values = ratios[mode]
        g = geo(values)
        med = statistics.median(values) if values else None
        faster = sum(v > 1 for v in values)
        summary[mode] = {
            "matched_cases": len(values), "geometric_mean_speedup": g,
            "median_speedup": med, "faster_cases": faster,
        }
        report.append(f"| {mode} | {len(values)} | {show(g)} | "
                      f"{show(med)} | {faster}/{len(values)} |")
    report += ["", "### By benchmark class", "",
               "| Class | V2 | V3 | V4 |", "|---|---:|---:|---:|"]
    for name, modes in sorted(grouped.items()):
        report.append("| " + name + " | " + " | ".join(
            show(geo(modes.get(mode, []))) for mode in TARGETS
        ) + " |")
    report += ["", "Full per-case measurements are included in the PocketFFT TSV artifact.",
               "Shared-runner frequency and scheduling variability remain possible.", ""]
    return "\n".join(report), "\n".join(table) + "\n", summary



def render_native_nd_comparison() -> tuple[str, str, dict]:
    """Compare native and legacy ND paths on identical V3 CPU dispatch."""
    native_file = artifact_path("X86_V3") / "asv-results.json"
    legacy_file = artifact_path("LEGACY_V3") / "asv-results.json"
    header = "benchmark\tparameters\tnative_v3_us\tlegacy_v3_us\tlegacy_over_native"
    if not native_file.exists() or not legacy_file.exists():
        return ("## Native versus legacy ND (V3)\n\nOne of the V3 profiles is missing.\n",
                header + "\n", {"status": "missing"})
    native = json.loads(native_file.read_text())
    legacy = json.loads(legacy_file.read_text())
    def group(payload):
        return {(r["benchmark"], tuple(r["parameters"])): r
                for r in payload["cases"]}
    nrows, lrows = group(native), group(legacy)
    rows = [header]
    grouped = {}
    for key in sorted(set(nrows) & set(lrows)):
        name, params = key
        if not name.startswith("bench_fft.FFTND."):
            continue
        n, l = nrows[key], lrows[key]
        tn, tl = n.get("asv_value"), l.get("asv_value")
        if (n.get("status") != "completed" or l.get("status") != "completed"
                or not isinstance(tn, (int, float))
                or not isinstance(tl, (int, float))
                or not math.isfinite(tn) or not math.isfinite(tl)
                or tn <= 0 or tl <= 0):
            continue
        ratio = float(tl) / float(tn)
        grouped.setdefault(name, []).append(ratio)
        rows.append("\t".join([
            name, ", ".join(params), f"{tn*1e6:.6f}",
            f"{tl*1e6:.6f}", f"{ratio:.6f}",
        ]))
    ratios = [x for values in grouped.values() for x in values]
    geo = math.exp(statistics.fmean(math.log(x) for x in ratios)) if ratios else None
    summary = {"matched_nd_cases": len(ratios), "geometric_mean_legacy_over_native": geo,
               "faster_native_cases": sum(x > 1 for x in ratios)}
    md = [
        "## Native versus legacy multi-axis FFT (both X86_V3)",
        "",
        "**Ratio = legacy-ND time / native-ND time**; greater than 1.00x favors native ND.",
        "Both variants use the same cache-disabled NumPy DUCC build. The legacy profile",
        "sets NPY_DUCCFFT_NATIVE_ND=0. Only complex FFTND cases are compared.",
        "",
        f"Matched ND cases: {len(ratios)}; geometric mean: "
        + (f"{geo:.3f}x" if geo is not None else "unavailable")
        + f"; native faster: {summary['faster_native_cases']}/{len(ratios)}.",
        "",
        "Full results in fft-native-nd-comparison.tsv.",
        "",
    ]
    return "\n".join(md), "\n".join(rows) + "\n", summary


def summarize() -> int:
    runner_temp = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    summary_file = Path(os.environ.get("GITHUB_STEP_SUMMARY", "/dev/null"))
    names = PROFILES
    rows = []
    all_passed = True

    for profile in names:
        path = runner_temp / f"fft-dispatch-{profile.lower()}" / "results.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {"status": "missing", "error": "profile report was not produced"}
        step_outcome = os.environ.get(f"{profile}_STEP_OUTCOME", "unknown")
        passed = data.get("status") == "passed" and step_outcome == "success"
        all_passed = all_passed and passed
        duration = data.get("benchmark_execution_seconds")
        duration = f"{duration}s" if duration is not None else "—"
        rows.append(
            "| {profile} | {status} | {selected} | {collected} | {completed} | "
            "{failed} | {skipped} | {duration} | {reason} |".format(
                profile=profile,
                status="passed" if passed else f"failed ({step_outcome})",
                selected=data.get("selected_target", "—"),
                collected=data.get("collected_cases", 0),
                completed=data.get("completed_cases", 0),
                failed=data.get("failed_cases", 0),
                skipped=data.get("skipped_cases", 0),
                duration=duration,
                reason=data.get("error", "—").replace("|", "\\|"),
            )
        )

    host_file = runner_temp / "fft-dispatch-diagnostics" / "host.json"
    try:
        host = json.loads(host_file.read_text())
        feature_groups = host.get("cpu_feature_groups", {})
        cpu = host.get("cpu_model", "unknown")
        arch = host.get("architecture", "unknown")
        host_summary = (
            f"CPU: {cpu} ({arch}); "
            + ", ".join(f"{name}={feature_groups.get(name, False)}" for name in TARGETS)
        )
    except (OSError, ValueError):
        host_summary = "Host diagnostics unavailable."
        all_passed = False

    build_outcome = os.environ.get("BUILD_STEP_OUTCOME", "unknown")
    test_outcome = os.environ.get("FFT_TEST_STEP_OUTCOME", "unknown")
    host_outcome = os.environ.get("HOST_STEP_OUTCOME", "unknown")
    install_outcome = os.environ.get("INSTALL_STEP_OUTCOME", "unknown")
    machine_outcome = os.environ.get("ASV_MACHINE_STEP_OUTCOME", "unknown")
    if any(value != "success" for value in (
        install_outcome, build_outcome, test_outcome, host_outcome, machine_outcome,
        os.environ.get("NATIVE_ND_TEST_STEP_OUTCOME", "unknown"),
        *(os.environ.get(key, "unknown") for key in (
            "POCKETFFT_CHECKOUT_OUTCOME", "POCKETFFT_CASES_OUTCOME",
            "POCKETFFT_BUILD_OUTCOME", "POCKETFFT_MACHINE_OUTCOME",
        )),
    )):
        all_passed = False

    content = "\n".join([
        "## FFT CPU dispatch ASV experiment",
        "",
        host_summary,
        "",
        "Install: {install}; build: {build}; DUCC FFT tests: {tests}; "
        "host report: {host}; ASV machine setup: {machine}.".format(
            install=os.environ.get("INSTALL_STEP_OUTCOME", "unknown"),
            build=build_outcome,
            tests=test_outcome,
            host=host_outcome,
            machine=os.environ.get("ASV_MACHINE_STEP_OUTCOME", "unknown"),
        ),
        "",
        "| Mode | Status | Selected target | ASV cases | Completed | Failures | Skips | Time | Detail |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "Non-quick ASV: warmup 0.15 s per round, 7 repeated samples, 2 rounds per case. Timings are medians of samples; calibration is managed by ASV.",
        "",
    ])
    perf_markdown, timing_tsv, timing_summary = render_performance_report()
    pocket_markdown, pocket_tsv, pocket_summary = render_pocketfft_comparison()
    native_markdown, native_tsv, native_summary = render_native_nd_comparison()
    content += "\n\n" + perf_markdown + "\n\n" + pocket_markdown + "\n\n" + native_markdown
    diagnostic_dir = artifact_path("diagnostics")
    (diagnostic_dir / "FFT_DISPATCH_REPORT.md").write_text(content + "\n")
    (diagnostic_dir / "fft-dispatch-timings.tsv").write_text(timing_tsv)
    write_json(diagnostic_dir / "fft-dispatch-comparison.json", timing_summary)
    (diagnostic_dir / "fft-pocketfft-comparison.tsv").write_text(pocket_tsv)
    write_json(diagnostic_dir / "fft-pocketfft-comparison.json", pocket_summary)
    (diagnostic_dir / "fft-native-nd-comparison.tsv").write_text(native_tsv)
    write_json(diagnostic_dir / "fft-native-nd-comparison.json", native_summary)
    print(content)
    if summary_file != Path("/dev/null"):
        with summary_file.open("a") as handle:
            handle.write(content + "\n")
    return 0 if all_passed else 1


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fft_dispatch_bench.py host | run X86_V2|X86_V3|X86_V4|POCKETFFT|LEGACY_V3 | summary", file=sys.stderr)
        return 2
    command = sys.argv[1]
    if command == "host":
        return host_profile()
    if command == "run" and len(sys.argv) == 3 and sys.argv[2] in PROFILES:
        return run_profile(sys.argv[2])
    if command == "summary":
        return summarize()
    print(f"invalid command: {' '.join(sys.argv[1:])}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
