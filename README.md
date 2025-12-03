# Rendering tests for rattler-build

The recipes in this repository are sourced from [conda-forge](https://github.com/conda-forge) and are used for testing the rattler-build recipe parser.

## Requirements

- Python 3 with `pyyaml`, `tomli` (or `tomllib` on Python 3.11+), and optionally `deepdiff`
- `rattler-build` installed and available in PATH (or specify path with `--rattler-build`)
- Git (for cloning feedstocks)

## Scripts

### generate_rendering_tests.py

Generates ground-truth test data by cloning conda-forge feedstocks and rendering them with rattler-build.

```bash
# Generate tests for 200 random feedstocks (default)
./generate_rendering_tests.py

# Generate tests for a specific number of feedstocks
./generate_rendering_tests.py --count 50

# Generate test for a specific feedstock
./generate_rendering_tests.py --feedstock numpy

# Use a specific rattler-build binary
./generate_rendering_tests.py --rattler-build /path/to/rattler-build

# Set random seed for reproducible selection
./generate_rendering_tests.py --seed 42

# Specify output directory
./generate_rendering_tests.py --output ./my-tests
```

### run_rendering_tests.py

Runs the test suite, comparing rattler-build output against the ground-truth data.

```bash
# Run all tests (sequential)
./run_rendering_tests.py

# Run tests in parallel (recommended for large test suites)
./run_rendering_tests.py --jobs 50

# Test a specific feedstock
./run_rendering_tests.py --feedstock numpy

# Stop on first failure
./run_rendering_tests.py --fail-fast

# Save diff files for failures
./run_rendering_tests.py --save-diffs ./diffs

# Save results to JSON
./run_rendering_tests.py --json-output results.json

# Re-run only previously failing tests
./run_rendering_tests.py --rerun-failures test_failures.json

# Use a specific rattler-build binary
./run_rendering_tests.py --rattler-build /path/to/rattler-build

# Verbose output
./run_rendering_tests.py --verbose
```

### render.sh

A convenience script for quickly rendering a single feedstock recipe.

```bash
# Render a feedstock (picks first variant)
./render.sh nwchem

# Render with a specific variant pattern
./render.sh nwchem linux*
./render.sh hf-xet "linux_64*"
```

## Directory Structure

```
rendering-tests/
├── <feedstock-name>/
│   ├── recipe/           # The recipe.yaml and supporting files
│   ├── variants/         # Variant YAML files from .ci_support
│   └── expected/         # Expected rattler-build --render-only output
│       ├── <variant>.json       # Expected JSON output
│       └── <variant>.meta.json  # Metadata (variant file, target platform)
└── generation_metadata.json     # Metadata about the generation run
```

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

The recipes contained in `rendering-tests/` are derived from conda-forge feedstocks, which are also licensed under BSD-3-Clause.