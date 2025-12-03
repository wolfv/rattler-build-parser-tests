#!/usr/bin/env -S pixi run python3
"""
Test rattler-build rendering against ground-truth test data.

This script:
1. Reads test data from the rendering-tests directory
2. Runs rattler-build --render-only on each recipe with its variant files
3. Compares the output against the expected ground-truth data
4. Reports any differences

Usage:
  ./run_rendering_tests.py                           # Test with local build
  ./run_rendering_tests.py --jobs 50                 # Run 50 tests in parallel
  ./run_rendering_tests.py --feedstock numpy         # Test specific feedstock
  ./run_rendering_tests.py --fail-fast               # Stop on first failure
"""

import argparse
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import difflib
import re
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading

@dataclass
class VariantTestResult:
    """Result of testing a single variant."""
    feedstock: str
    variant: str
    target_platform: Optional[str]
    passed: bool
    reason: str
    error: Optional[str] = None
    details: Optional[str] = None

# Lock for thread-safe printing
print_lock = threading.Lock()

try:
    from deepdiff import DeepDiff
except ImportError:
    DeepDiff = None
    print("Warning: deepdiff not installed, using basic comparison")

try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: pyyaml not installed, context variable handling disabled")


# Default test data directory
RENDERING_TESTS_DIR = Path(__file__).parent / "rendering-tests"


def get_context_keys(recipe_path: Path) -> Set[str]:
    """Extract context variable keys from a recipe.yaml file."""
    if yaml is None:
        return set()

    recipe_yaml = recipe_path / "recipe.yaml"
    if not recipe_yaml.exists():
        return set()

    try:
        with open(recipe_yaml, 'r') as f:
            content = f.read()

        # Parse YAML (might fail on Jinja templates, but context section is usually simple)
        recipe = yaml.safe_load(content)
        if recipe and isinstance(recipe, dict) and 'context' in recipe:
            context = recipe['context']
            if isinstance(context, dict):
                return set(context.keys())
    except Exception:
        pass

    return set()


def compute_hash_from_variant(variant: Dict) -> str:
    """
    Compute the hash from a variant dictionary using the same algorithm as rattler-build.

    The hash is computed by:
    1. JSON serializing the variant (sorted keys, Python-style formatting)
    2. Computing SHA1 hash
    3. Taking first 7 characters
    """
    # Sort keys (BTreeMap behavior)
    sorted_variant = dict(sorted(variant.items()))

    # JSON serialize with Python-style formatting (", " separator)
    # We need to match the exact format rattler-build uses
    json_str = json.dumps(sorted_variant, separators=(', ', ': '), sort_keys=True)

    # SHA1 hash and take first 7 chars
    hash_bytes = hashlib.sha1(json_str.encode('utf-8')).hexdigest()
    return hash_bytes[:7]


def adjust_expected_for_context_overrides(
    expected: Dict,
    actual: Dict,
    context_keys: Set[str]
) -> Dict:
    """
    Adjust expected output when context variables override variant keys.

    If a variant key exists in expected but not in actual, and that key is defined
    in the recipe's context section, then we need to recompute the expected hash
    with that key removed.
    """
    if not context_keys:
        return expected

    adjusted = copy.deepcopy(expected)

    for i, output in enumerate(adjusted):
        if 'build_configuration' not in output:
            continue

        build_config = output['build_configuration']
        expected_variant = build_config.get('variant', {})
        actual_variant = actual[i]['build_configuration'].get('variant', {}) if i < len(actual) else {}

        # Find keys that are in expected but not in actual
        missing_keys = set(expected_variant.keys()) - set(actual_variant.keys())

        # Check if these missing keys are context variables
        context_overridden_keys = missing_keys & context_keys

        if not context_overridden_keys:
            continue

        # Remove context-overridden keys from the expected variant
        new_variant = {k: v for k, v in expected_variant.items() if k not in context_overridden_keys}

        # Recompute the hash
        new_hash = compute_hash_from_variant(new_variant)

        # Get the original build string to extract the pattern
        old_hash = build_config.get('hash', {}).get('hash', '')
        old_build_string = output.get('recipe', {}).get('build', {}).get('string', '')

        # Update the variant
        build_config['variant'] = new_variant

        # Update the hash
        if 'hash' in build_config:
            build_config['hash']['hash'] = new_hash

        # Update the build string (replace old hash with new hash)
        if old_hash and old_build_string and old_hash in old_build_string:
            new_build_string = old_build_string.replace(old_hash, new_hash)
            if 'recipe' in output and 'build' in output['recipe']:
                output['recipe']['build']['string'] = new_build_string

        # Update subpackages build strings
        if 'subpackages' in build_config:
            for pkg_name, pkg_data in build_config['subpackages'].items():
                if 'build_string' in pkg_data and old_hash in pkg_data['build_string']:
                    pkg_data['build_string'] = pkg_data['build_string'].replace(old_hash, new_hash)

    return adjusted


def run_rattler_build(
    rattler_build_cmd: str,
    recipe_path: Path,
    variant_file: Optional[Path] = None,
    target_platform: Optional[str] = None
) -> Tuple[bool, str]:
    """Run rattler-build with --render-only and return success status and JSON output."""
    recipe_yaml = recipe_path / "recipe.yaml"
    if not recipe_yaml.exists():
        return False, f"recipe.yaml not found in {recipe_path}"

    cmd = [rattler_build_cmd, "build", "--no-build-id", "--recipe", str(recipe_yaml), "--render-only"]

    if variant_file:
        cmd.extend(["-m", str(variant_file)])

    if target_platform:
        cmd.extend(["--target-platform", target_platform])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def normalize_json_output(output: str) -> Dict:
    """Parse and normalize JSON output."""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        lines = output.strip().split('\n')
        for line in lines:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise ValueError("Could not parse JSON from output")


def compare_outputs(
    expected: Dict,
    actual: Dict,
    feedstock_name: str,
    variant_name: str,
    output_dir: Optional[Path] = None
) -> Tuple[bool, str]:
    """Compare expected and actual JSON outputs."""
    # Keys to exclude from comparison (expected to differ between runs)
    exclude_regex_paths = [
        r"root\[\d+\]\['build_configuration'\]\['timestamp'\]",
        r"root\[\d+\]\['system_tools'\]\['rattler-build'\]",
    ]

    if expected == actual:
        return True, "Outputs match exactly"

    if DeepDiff:
        deep_diff = DeepDiff(
            expected,
            actual,
            ignore_order=False,
            verbose_level=2,
            exclude_regex_paths=exclude_regex_paths
        )

        if not deep_diff:
            return True, "Outputs match (ignoring timestamp and rattler-build version)"

        # Generate diff message
        diff_msg = "Outputs differ:\n"
        diff_msg += deep_diff.pretty()

        # Save diff files if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_variant = variant_name.replace('/', '_').replace('\\', '_')
            base_name = f"{feedstock_name}_{safe_variant}"

            # Save expected and actual JSON
            with open(output_dir / f"{base_name}_expected.json", 'w') as f:
                json.dump(expected, f, indent=2, sort_keys=True)
            with open(output_dir / f"{base_name}_actual.json", 'w') as f:
                json.dump(actual, f, indent=2, sort_keys=True)

            # Save unified diff
            expected_str = json.dumps(expected, indent=2, sort_keys=True)
            actual_str = json.dumps(actual, indent=2, sort_keys=True)
            unified_diff = list(difflib.unified_diff(
                expected_str.splitlines(keepends=True),
                actual_str.splitlines(keepends=True),
                fromfile='expected.json',
                tofile='actual.json'
            ))
            with open(output_dir / f"{base_name}_diff.txt", 'w') as f:
                f.write(''.join(unified_diff))
                f.write("\n\n=== DEEPDIFF OUTPUT ===\n")
                f.write(deep_diff.pretty())

            diff_msg += f"\n\nDiff files saved to: {output_dir / base_name}_*"

        return False, diff_msg
    else:
        # Basic comparison without deepdiff
        expected_str = json.dumps(expected, indent=2, sort_keys=True)
        actual_str = json.dumps(actual, indent=2, sort_keys=True)

        if expected_str == actual_str:
            return True, "Outputs match"

        # Generate unified diff
        unified_diff = list(difflib.unified_diff(
            expected_str.splitlines(keepends=True),
            actual_str.splitlines(keepends=True),
            fromfile='expected.json',
            tofile='actual.json'
        ))

        diff_preview = ''.join(unified_diff[:100])
        if len(unified_diff) > 100:
            diff_preview += f"\n... ({len(unified_diff) - 100} more lines)"

        return False, f"Outputs differ:\n{diff_preview}"


def get_feedstock_dirs(tests_dir: Path) -> List[Path]:
    """Get all feedstock directories in the tests directory."""
    feedstocks = []
    for item in tests_dir.iterdir():
        if item.is_dir() and (item / "recipe").exists():
            feedstocks.append(item)
    return sorted(feedstocks)


@dataclass
class TestCase:
    """A single test case to run."""
    feedstock_dir: Path
    feedstock_name: str
    variant_name: str
    variant_file: Optional[Path]
    target_platform: Optional[str]
    expected_file: Path
    context_keys: Set[str]


def collect_test_cases(tests_dir: Path, feedstock_filter: Optional[str] = None) -> List[TestCase]:
    """Collect all test cases from the test directory."""
    test_cases = []

    if feedstock_filter:
        feedstock_dir = tests_dir / feedstock_filter
        if not feedstock_dir.exists():
            return []
        feedstock_dirs = [feedstock_dir]
    else:
        feedstock_dirs = get_feedstock_dirs(tests_dir)

    for feedstock_dir in feedstock_dirs:
        feedstock_name = feedstock_dir.name
        recipe_path = feedstock_dir / "recipe"
        expected_dir = feedstock_dir / "expected"
        variants_dir = feedstock_dir / "variants"

        if not recipe_path.exists() or not expected_dir.exists():
            continue

        # Get context keys from the recipe
        context_keys = get_context_keys(recipe_path)

        # Find all expected output files
        expected_files = list(expected_dir.glob("*.json"))
        expected_files = [f for f in expected_files if not f.name.endswith('.meta.json')]

        for expected_file in expected_files:
            variant_base_name = expected_file.stem
            meta_file = expected_dir / f"{variant_base_name}.meta.json"

            variant_file = None
            target_platform = None

            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get('variant_file'):
                            variant_file = variants_dir / metadata['variant_file']
                            if not variant_file.exists():
                                variant_file_yaml = variants_dir / (metadata['variant_file'] + '.yaml')
                                if variant_file_yaml.exists():
                                    variant_file = variant_file_yaml
                                else:
                                    variant_file = None
                        target_platform = metadata.get('target_platform')
                except Exception:
                    pass

            variant_name = variant_file.name if variant_file else "no-variant"

            test_cases.append(TestCase(
                feedstock_dir=feedstock_dir,
                feedstock_name=feedstock_name,
                variant_name=variant_name,
                variant_file=variant_file,
                target_platform=target_platform,
                expected_file=expected_file,
                context_keys=context_keys,
            ))

    return test_cases


def load_failures_filter(failures_file: Path) -> Set[Tuple[str, str]]:
    """Load a failures file and return a set of (feedstock, variant) tuples to filter by."""
    if not failures_file.exists():
        print(f"Warning: Failures file not found: {failures_file}")
        return set()

    try:
        with open(failures_file, 'r') as f:
            data = json.load(f)

        failures = set()
        for item in data.get('failures', []):
            feedstock = item.get('feedstock')
            variant = item.get('variant')
            if feedstock and variant:
                failures.add((feedstock, variant))

        return failures
    except Exception as e:
        print(f"Warning: Could not load failures file: {e}")
        return set()


def filter_test_cases_by_failures(test_cases: List[TestCase], failures: Set[Tuple[str, str]]) -> List[TestCase]:
    """Filter test cases to only include those in the failures set."""
    return [tc for tc in test_cases if (tc.feedstock_name, tc.variant_name) in failures]


def save_failures(failures: List[Dict], output_file: Path):
    """Save failing test cases to a JSON file for later re-running."""
    # Strip details for the JSON file (keep it small for re-running)
    compact_failures = []
    for f in failures:
        compact_failures.append({
            'feedstock': f['feedstock'],
            'variant': f['variant'],
            'target_platform': f.get('target_platform'),
            'reason': f['reason']
        })

    data = {
        'failures': compact_failures,
        'count': len(compact_failures)
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(compact_failures)} failing test(s) to: {output_file}")


def save_failures_markdown(failures: List[Dict], output_file: Path):
    """Save detailed failure information to a markdown file for human/agent consumption."""
    with open(output_file, 'w') as f:
        f.write("# Test Failures Report\n\n")
        f.write(f"Total failures: {len(failures)}\n\n")

        # Group by feedstock
        by_feedstock: Dict[str, List[Dict]] = {}
        for failure in failures:
            fs = failure['feedstock']
            if fs not in by_feedstock:
                by_feedstock[fs] = []
            by_feedstock[fs].append(failure)

        f.write("## Summary\n\n")
        f.write("| Feedstock | Failures |\n")
        f.write("|-----------|----------|\n")
        for fs in sorted(by_feedstock.keys()):
            f.write(f"| {fs} | {len(by_feedstock[fs])} |\n")
        f.write("\n")

        f.write("## Detailed Failures\n\n")
        for fs in sorted(by_feedstock.keys()):
            f.write(f"### {fs}\n\n")
            for failure in by_feedstock[fs]:
                variant = failure['variant']
                platform = failure.get('target_platform', 'N/A')
                reason = failure['reason']
                error = failure.get('error', '')
                details = failure.get('details', '')

                f.write(f"#### {variant}\n\n")
                f.write(f"- **Platform**: {platform}\n")
                f.write(f"- **Reason**: {reason}\n")

                if error:
                    f.write(f"\n**Error**:\n```\n{error}\n```\n")

                if details:
                    # Truncate very long details
                    if len(details) > 5000:
                        details = details[:5000] + "\n... (truncated)"
                    f.write(f"\n**Details**:\n```\n{details}\n```\n")

                f.write("\n")

    print(f"Saved detailed failure report to: {output_file}")


def test_single_variant(
    test_case: TestCase,
    rattler_build_cmd: str,
    diff_output_dir: Optional[Path] = None,
    verbose: bool = False
) -> VariantTestResult:
    """Test a single variant and return the result."""
    recipe_path = test_case.feedstock_dir / "recipe"

    # Load expected output
    try:
        with open(test_case.expected_file, 'r') as f:
            expected_output = json.load(f)
    except Exception as e:
        return VariantTestResult(
            feedstock=test_case.feedstock_name,
            variant=test_case.variant_name,
            target_platform=test_case.target_platform,
            passed=False,
            reason='expected_load_error',
            error=str(e)
        )

    # Run rattler-build
    success, output = run_rattler_build(
        rattler_build_cmd,
        recipe_path,
        test_case.variant_file,
        test_case.target_platform
    )

    if not success:
        return VariantTestResult(
            feedstock=test_case.feedstock_name,
            variant=test_case.variant_name,
            target_platform=test_case.target_platform,
            passed=False,
            reason='render_failed',
            error=output[:500]
        )

    # Parse actual output
    try:
        actual_output = normalize_json_output(output)
    except Exception as e:
        return VariantTestResult(
            feedstock=test_case.feedstock_name,
            variant=test_case.variant_name,
            target_platform=test_case.target_platform,
            passed=False,
            reason='parse_error',
            error=str(e)
        )

    # Adjust expected output for context variable overrides
    adjusted_expected = adjust_expected_for_context_overrides(
        expected_output,
        actual_output,
        test_case.context_keys
    )

    # Compare outputs
    match, message = compare_outputs(
        adjusted_expected,
        actual_output,
        test_case.feedstock_name,
        test_case.variant_name,
        diff_output_dir
    )

    if match:
        return VariantTestResult(
            feedstock=test_case.feedstock_name,
            variant=test_case.variant_name,
            target_platform=test_case.target_platform,
            passed=True,
            reason='passed'
        )
    else:
        return VariantTestResult(
            feedstock=test_case.feedstock_name,
            variant=test_case.variant_name,
            target_platform=test_case.target_platform,
            passed=False,
            reason='output_mismatch',
            details=message[:2000]
        )


def test_feedstock(
    feedstock_dir: Path,
    rattler_build_cmd: str,
    diff_output_dir: Optional[Path] = None,
    verbose: bool = False
) -> Dict:
    """Test a single feedstock against its expected outputs."""
    feedstock_name = feedstock_dir.name
    print(f"\nTesting feedstock: {feedstock_name}")
    print("-" * 60)

    result = {
        'feedstock': feedstock_name,
        'variants_tested': 0,
        'variants_passed': 0,
        'variants_failed': [],
        'errors': []
    }

    recipe_path = feedstock_dir / "recipe"
    expected_dir = feedstock_dir / "expected"
    variants_dir = feedstock_dir / "variants"

    if not recipe_path.exists():
        result['errors'].append("No recipe directory found")
        return result

    if not expected_dir.exists():
        result['errors'].append("No expected outputs directory found")
        return result

    # Get context keys from the recipe for adjusting expected outputs
    context_keys = get_context_keys(recipe_path)

    # Find all expected output files
    expected_files = list(expected_dir.glob("*.json"))
    # Filter out metadata files
    expected_files = [f for f in expected_files if not f.name.endswith('.meta.json')]

    if not expected_files:
        result['errors'].append("No expected output files found")
        return result

    print(f"  Found {len(expected_files)} expected outputs")

    for expected_file in expected_files:
        variant_base_name = expected_file.stem
        meta_file = expected_dir / f"{variant_base_name}.meta.json"

        # Load metadata
        variant_file = None
        target_platform = None

        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('variant_file'):
                        variant_file = variants_dir / metadata['variant_file']
                        if not variant_file.exists():
                            # Try adding .yaml extension if not present
                            variant_file_yaml = variants_dir / (metadata['variant_file'] + '.yaml')
                            if variant_file_yaml.exists():
                                variant_file = variant_file_yaml
                            else:
                                print(f"    Warning: Variant file not found: {variant_file}")
                                variant_file = None
                    target_platform = metadata.get('target_platform')
            except Exception as e:
                print(f"    Warning: Could not read metadata: {e}")

        variant_name = variant_file.name if variant_file else "no-variant"
        print(f"  Testing variant: {variant_name}", end="")
        if target_platform:
            print(f" (target: {target_platform})", end="")
        print()

        result['variants_tested'] += 1

        # Load expected output
        try:
            with open(expected_file, 'r') as f:
                expected_output = json.load(f)
        except Exception as e:
            print(f"    ✗ Failed to load expected output: {e}")
            result['variants_failed'].append({
                'variant': variant_name,
                'reason': 'expected_load_error',
                'error': str(e)
            })
            continue

        # Run rattler-build
        success, output = run_rattler_build(
            rattler_build_cmd,
            recipe_path,
            variant_file,
            target_platform
        )

        if not success:
            print(f"    ✗ Rendering failed: {output[:200]}")
            result['variants_failed'].append({
                'variant': variant_name,
                'reason': 'render_failed',
                'error': output[:500]
            })
            continue

        # Parse actual output
        try:
            actual_output = normalize_json_output(output)
        except Exception as e:
            print(f"    ✗ Failed to parse output: {e}")
            result['variants_failed'].append({
                'variant': variant_name,
                'reason': 'parse_error',
                'error': str(e)
            })
            continue

        # Adjust expected output for context variable overrides
        # When a variant key is defined in the recipe's context section, it shouldn't
        # be in the variant dictionary (the new parser correctly excludes it)
        adjusted_expected = adjust_expected_for_context_overrides(
            expected_output,
            actual_output,
            context_keys
        )

        # Compare outputs
        match, message = compare_outputs(
            adjusted_expected,
            actual_output,
            feedstock_name,
            variant_name,
            diff_output_dir
        )

        if match:
            print(f"    ✓ Passed")
            result['variants_passed'] += 1
        else:
            if verbose:
                print(f"    ✗ Failed: {message}")
            else:
                # Print truncated message
                lines = message.split('\n')
                preview = '\n'.join(lines[:10])
                if len(lines) > 10:
                    preview += f"\n    ... ({len(lines) - 10} more lines)"
                print(f"    ✗ Failed:\n    {preview}")
            result['variants_failed'].append({
                'variant': variant_name,
                'reason': 'output_mismatch',
                'details': message[:2000]
            })

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test rattler-build rendering against ground-truth data"
    )
    parser.add_argument(
        "--rattler-build", "-r",
        type=str,
        default=str(Path(__file__).parent / "target" / "release" / "rattler-build"),
        help="Path to rattler-build command to test"
    )
    parser.add_argument(
        "--tests-dir", "-t",
        type=str,
        default=str(RENDERING_TESTS_DIR),
        help=f"Directory containing test data (default: {RENDERING_TESTS_DIR})"
    )
    parser.add_argument(
        "--feedstock", "-f",
        type=str,
        help="Test only a specific feedstock"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full diff output"
    )
    parser.add_argument(
        "--save-diffs", "-s",
        type=str,
        help="Directory to save diff files for failures"
    )
    parser.add_argument(
        "--json-output", "-j",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--jobs", "-J",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, use 50 for parallel execution)"
    )
    parser.add_argument(
        "--save-failures",
        type=str,
        default="test_failures.json",
        help="File to save failing test cases (default: test_failures.json)"
    )
    parser.add_argument(
        "--rerun-failures",
        type=str,
        help="Re-run only the failing tests from this JSON file"
    )

    args = parser.parse_args()

    tests_dir = Path(args.tests_dir)
    rattler_build_cmd = args.rattler_build
    diff_output_dir = Path(args.save_diffs) if args.save_diffs else None

    print("=" * 80)
    print("Rattler-Build Rendering Test Suite")
    print("=" * 80)

    # Verify rattler-build command exists
    if not Path(rattler_build_cmd).exists() and not shutil.which(rattler_build_cmd):
        print(f"Error: Could not find rattler-build: {rattler_build_cmd}")
        sys.exit(1)

    try:
        result = subprocess.run(
            [rattler_build_cmd, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Testing rattler-build: {rattler_build_cmd}")
        print(f"Version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: Could not run rattler-build: {e}")
        sys.exit(1)

    # Verify tests directory exists
    if not tests_dir.exists():
        print(f"Error: Tests directory not found: {tests_dir}")
        print("Run generate_rendering_tests.py first to create test data.")
        sys.exit(1)

    print(f"Tests directory: {tests_dir}")
    print(f"Parallel jobs: {args.jobs}")
    if diff_output_dir:
        print(f"Saving diffs to: {diff_output_dir}")

    # Get feedstocks to test
    if args.feedstock:
        feedstock_dir = tests_dir / args.feedstock
        if not feedstock_dir.exists():
            print(f"Error: Feedstock not found: {args.feedstock}")
            sys.exit(1)
        feedstock_dirs = [feedstock_dir]
    else:
        feedstock_dirs = get_feedstock_dirs(tests_dir)

    print(f"Feedstocks to test: {len(feedstock_dirs)}")

    # Run tests
    all_results = []
    failed = False

    # Track all failures for saving
    all_failures: List[Dict] = []

    if args.jobs > 1:
        # Parallel execution mode
        test_cases = collect_test_cases(tests_dir, args.feedstock)

        # Filter by previous failures if requested
        if args.rerun_failures:
            failures_filter = load_failures_filter(Path(args.rerun_failures))
            if failures_filter:
                original_count = len(test_cases)
                test_cases = filter_test_cases_by_failures(test_cases, failures_filter)
                print(f"Re-running {len(test_cases)} previously failing tests (from {original_count} total)")
            else:
                print("No failures to re-run, running all tests")

        print(f"Total variants to test: {len(test_cases)}")
        print(f"Running with {args.jobs} parallel jobs...")
        print()

        results_by_feedstock: Dict[str, Dict] = {}
        completed = 0
        stop_flag = threading.Event()

        def run_test(tc: TestCase) -> VariantTestResult:
            if stop_flag.is_set():
                return VariantTestResult(
                    feedstock=tc.feedstock_name,
                    variant=tc.variant_name,
                    target_platform=tc.target_platform,
                    passed=False,
                    reason='skipped',
                    error='Stopped due to --fail-fast'
                )
            return test_single_variant(tc, rattler_build_cmd, diff_output_dir, args.verbose)

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_tc = {executor.submit(run_test, tc): tc for tc in test_cases}

            for future in as_completed(future_to_tc):
                tc = future_to_tc[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = VariantTestResult(
                        feedstock=tc.feedstock_name,
                        variant=tc.variant_name,
                        target_platform=tc.target_platform,
                        passed=False,
                        reason='exception',
                        error=str(e)
                    )

                completed += 1

                # Update feedstock results
                if result.feedstock not in results_by_feedstock:
                    results_by_feedstock[result.feedstock] = {
                        'feedstock': result.feedstock,
                        'variants_tested': 0,
                        'variants_passed': 0,
                        'variants_failed': [],
                        'errors': []
                    }

                fs_result = results_by_feedstock[result.feedstock]
                fs_result['variants_tested'] += 1

                if result.passed:
                    fs_result['variants_passed'] += 1
                    status = "✓"
                else:
                    failure_info = {
                        'variant': result.variant,
                        'reason': result.reason,
                        'error': result.error,
                        'details': result.details
                    }
                    fs_result['variants_failed'].append(failure_info)
                    # Also track for saving to failures file (with full details)
                    all_failures.append({
                        'feedstock': result.feedstock,
                        'variant': result.variant,
                        'target_platform': result.target_platform,
                        'reason': result.reason,
                        'error': result.error,
                        'details': result.details
                    })
                    status = "✗"
                    failed = True

                    if args.fail_fast:
                        stop_flag.set()

                # Thread-safe printing
                with print_lock:
                    platform_info = f" ({result.target_platform})" if result.target_platform else ""
                    print(f"[{completed}/{len(test_cases)}] {status} {result.feedstock}/{result.variant}{platform_info}")
                    if not result.passed and args.verbose and result.details:
                        print(f"         {result.details[:200]}")

        all_results = list(results_by_feedstock.values())

    else:
        # Sequential execution mode (original behavior)
        for i, feedstock_dir in enumerate(feedstock_dirs, 1):
            print(f"\n[{i}/{len(feedstock_dirs)}]")
            result = test_feedstock(
                feedstock_dir,
                rattler_build_cmd,
                diff_output_dir,
                args.verbose
            )
            all_results.append(result)

            # Collect failures for saving (with full details)
            for failure in result['variants_failed']:
                all_failures.append({
                    'feedstock': result['feedstock'],
                    'variant': failure.get('variant', 'unknown'),
                    'target_platform': None,  # Not tracked in sequential mode
                    'reason': failure.get('reason', 'unknown'),
                    'error': failure.get('error'),
                    'details': failure.get('details')
                })

            if result['variants_failed'] or result['errors']:
                failed = True
                if args.fail_fast:
                    print("\n⚠ Stopping due to --fail-fast")
                    break

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tested = sum(r['variants_tested'] for r in all_results)
    total_passed = sum(r['variants_passed'] for r in all_results)
    total_failed = total_tested - total_passed
    feedstocks_passed = len([r for r in all_results if r['variants_tested'] > 0 and not r['variants_failed'] and not r['errors']])
    feedstocks_failed = len([r for r in all_results if r['variants_failed'] or r['errors']])

    print(f"Feedstocks tested: {len(all_results)}")
    print(f"Feedstocks passed: {feedstocks_passed}")
    print(f"Feedstocks failed: {feedstocks_failed}")
    print(f"Total variants tested: {total_tested}")
    print(f"Total variants passed: {total_passed}")
    print(f"Total variants failed: {total_failed}")
    print(f"Pass rate: {100 * total_passed / total_tested if total_tested > 0 else 0:.1f}%")

    # List failures
    if total_failed > 0:
        print("\nFailed feedstocks:")
        for result in all_results:
            if result['variants_failed'] or result['errors']:
                print(f"\n  {result['feedstock']}:")
                if result['errors']:
                    for error in result['errors']:
                        print(f"    Error: {error}")
                for failure in result['variants_failed']:
                    reason = failure.get('reason', 'unknown')
                    variant = failure.get('variant', 'unknown')
                    print(f"    - {variant}: {reason}")

    # Save results to JSON if requested
    if args.json_output:
        output_file = Path(args.json_output)
        summary = {
            'feedstocks_tested': len(all_results),
            'feedstocks_passed': feedstocks_passed,
            'feedstocks_failed': feedstocks_failed,
            'variants_tested': total_tested,
            'variants_passed': total_passed,
            'variants_failed': total_failed,
            'pass_rate': total_passed / total_tested if total_tested > 0 else 0,
            'results': all_results
        }
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Save failures to file for re-running
    if all_failures and args.save_failures:
        save_failures(all_failures, Path(args.save_failures))
        # Also save detailed markdown report
        md_file = Path(args.save_failures).with_suffix('.md')
        save_failures_markdown(all_failures, md_file)

    # Exit with appropriate code
    if total_failed > 0:
        print(f"\n✗ {total_failed} variant(s) failed")
        sys.exit(1)
    else:
        # Remove failures file if all tests passed
        if args.save_failures:
            failures_path = Path(args.save_failures)
            if failures_path.exists():
                failures_path.unlink()
                print(f"Removed previous failures file: {failures_path}")
        print(f"\n✓ All {total_passed} variants passed!")
        sys.exit(0)


# Need to import shutil for which()
import shutil


if __name__ == "__main__":
    main()
