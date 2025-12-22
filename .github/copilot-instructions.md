## Purpose

Be concise and actionable. This repository implements an academic-grade financial planning
and valuation engine (cash flows, cost of capital, and valuation methods) that avoids
iterative circularity by using analytical formulas. Your suggestions or code changes
must preserve the analytical approach and the numeric consistency checks embedded in
the core modules.

## Big picture (what matters)

- Core pieces live under `src/financial_planning/core/`:
  - `cash_flow.py` — data classes (`CashFlowComponents`) and `CashFlowCalculator` with
    explicit consistency checks (e.g. CCF == CFE + CFD). Use these classes for any
    cash-flow related logic.
  - `circularity_solver.py` — analytical, closed-form solver for the circularity
    problem. Prefer using it over iterative WACC/Ke loops. It exposes `CircularitySolver`
    and `CircularityResults` dataclass.
  - `cost_of_capital.py` — helper formulas (Ku, Ke without circularity, tax shield PV).
  - `valuation.py` — `ValuationEngine` that wires the pieces together (APV, CCF, WACC,
    Equity/CFE). This is the integration point for valuation scenarios.

## Key patterns and conventions

- Inputs/outputs use dataclasses: `CashFlowComponents`, `ValuationInputs`, `ValuationResults`.
  Maintain clear shapes (lists, floats) and validation inside constructors or `_validate_*`
  methods.
- Validation is explicit and exception-driven — functions raise `ValueError` for
  invalid inputs or numeric inconsistency (e.g. the check in `calculate_all_cash_flows`).
  Preserve these checks when refactoring or adding features.
- Discount-rate choice for tax-shields is explicit: use the `psi` / `discount_rate_ts`
  parameter with values `'Ku'` or `'Kd'`. Several formulas branch on that option.
- Terminal value and tax-shield math are central — changes here affect all valuation
  methods. If modifying `calculate_terminal_value`, update both `circularity_solver.py`
  and `valuation.py` usage sites and run numerical checks.

## Developer workflows (how to run & test locally)

- Install dependencies used by the project:

  pip install -r docs/requirements.txt

- Quick smoke runs (each core module has an `example_usage` guarded by `if __name__ == "__main__"`):

  python -m financial_planning.core.cash_flow
  python -m financial_planning.core.circularity_solver
  python -m financial_planning.core.valuation

  Running these from the repository root ensures package imports (they rely on the
  package `financial_planning`). Use `python -m` so relative imports resolve.

- Tests: repository contains `tests/` — run the test suite with pytest after installing
  `pytest` from `docs/requirements.txt`:

  pytest -q

## What to look for when making changes

- Numeric equivalence: many methods implement several valuation approaches that
  should produce near-identical firm values (APV, CCF, WACC, Equity/CFE). When
  changing formulas, add a targeted unit test that computes results from `ValuationEngine.valuation_all_methods`
  and asserts small relative differences (tolerance ~1e-3 or project default).
- Preserve datatypes: functions often return primitives, dicts, or pandas DataFrames.
  Keep signatures stable or provide thin adapters.
- Errors are surfaced early: when adding new code paths, follow the existing pattern of
  validating inputs and raising `ValueError` with a clear message rather than returning
  None or silently correcting values.

## Files to reference for examples

- `src/financial_planning/core/cash_flow.py` — example of dataclass inputs, calculation flow,
  and the consistency check in `calculate_all_cash_flows`.
- `src/financial_planning/core/circularity_solver.py` — canonical analytical formulas,
  `psi` handling, and `calculate_terminal_value` logic.
- `src/financial_planning/core/valuation.py` — integration: how APV/CCF/WACC/CFE are composed
  and how terminal values are wired into valuations.
- `src/financial_planning/__init__.py` — top-level exports; update if you add a new public
  component to keep API stable.

## Avoid

- Introducing iterative heuristics that silently replace the analytical closed-form
  solutions. If an iterative method is added for experimentation, keep it separate
  and clearly labeled (e.g., `*_iterative.py`) and include tests that compare it to
  the analytical baseline.

## If something is unclear

- Point the reviewer to a small reproduction: the minimal `ValuationInputs` (4 periods)
  used by `valuation.example_usage()` is a good starting point. Include exact inputs
  and the expected numeric output (from `master`) when requesting help debugging
  numeric differences.

---
If you want, I can also add a short CONTRIBUTING section or example unit tests that
assert cross-method consistency for `ValuationEngine`. Tell me what style/tolerance
you prefer for numeric comparisons and I’ll add them.
