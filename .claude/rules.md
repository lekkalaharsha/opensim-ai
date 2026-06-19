# OpenQSim Coding Rules

## Rule 1: Interface First
Every backend MUST implement `QuantumSimulatorBackend` ABC.

## Rule 2: Immutable Raw Data
Once written to `data/raw/`, a benchmark JSON is NEVER modified.

## Rule 3: Schema Version on Every Record
Every JSON output must include `"schema_version": "0.1.0"`.

## Rule 4: Environment Metadata on Every Record
Every benchmark JSON must include the full `environment` block.

## Rule 5: Three-Part Timing
Timing ALWAYS recorded as compilation, execution, total.

## Rule 6: Fidelity Is Optional
`fidelity` must be `null` when not computable. No placeholders.

## Rule 7: OOM Is Not a Crash
`MemoryError` caught, recorded as `success=false`.

## Rule 8: Seeds Everywhere
Every circuit generator accepts a `seed` parameter.

## Rule 9: No Circular Imports
`backend/` never imports from `benchmark/`.

## Rule 10: GPU Memory Is Polled, Not Estimated
Use `pynvml` in a separate thread at 100ms intervals.

## Rule 11: Type Hints Are Mandatory
Every function signature has type hints.

## Rule 12: Docstrings Are Mandatory
Every public function has a Google-style docstring.

## Rule 13: Commit Convention
`feat(module):`, `fix(module):`, `test(module):`, `docs:`, `chore:`

## Rule 14: No Dead Code
No commented-out blocks. Remove, don't comment.

## Rule 15: Kaggle Checkpointing
Every 10 records: flush to disk. Every 50 records: upload artifact.