# Roadmap

This is an honest snapshot of where Smart CSV Toolkit actually is, based on
what's merged on `main` vs. what's still tracked as open issues. It's not a
committed timeline — just a sense of direction, kept in sync as issues open
and close.

## Now (what's actually built, on `main` today)

- **Guided CSV cleaning** (Tab 1) — config-driven steps from `table_steps.json`
  (impute, normalize/scale, binarize, remove outliers, NaN-threshold column/row
  drops, one-hot encoding, SMOTE/oversample/undersample, dataset multiplier,
  add noise), dispatched through `pipeline_logic.run_pipeline`.
- **LLM-assisted cleaning** — suggested steps, free-text instructions turned
  into executable Python, and a decision-tree-guided flow in Tab 2 — all
  routed through OpenRouter (`call_llm`) and executed only inside
  `safe_exec.run_sandboxed`: AST-validated, restricted-builtins code run in an
  isolated subprocess with wall-clock/CPU/memory limits (#4, #19).
- **Metadata inference** (Tab 2) — heuristic + LLM-fallback column-type
  detection (`metadata_inference.infer_column_type`), type-aware
  visualizations, and a **data-quality scoring report**
  (`generate_quality_report`, #18) that now correctly scores the live Tab 2
  working copy instead of the original upload (#22).
- **Recipes** — cleaning pipelines can be exported as replayable JSON and
  re-applied to a new CSV without reconfiguring checkboxes (#16), with
  malformed/incompatible recipe JSON now failing gracefully instead of
  crashing the app (#21).
- **Batch cleaning** — one recipe or one set of checked steps applied to many
  uploaded CSVs at once, with a per-file success/failure summary and a single
  zip download (#17).
- **Security hardening** — CSV formula/DDE injection escaping on export (#11),
  SSRF-guarded URL fetching for image/link/webpage previews (#8), stored-XSS
  fix in file-link rendering (#20), TLS verification restored on LLM calls
  (#6), and the sandboxing above (#4, #19).
- **Correctness fixes** — dtype corruption from `most_frequent` imputation,
  outlier removal wiping all-NaN columns, Normalize/Scale/Binarize giving
  sklearn stack traces instead of clear errors on non-numeric data (#23, #24),
  bounded audit-log writes (#10), per-session isolated memory store (#9).
- **Operational basics** — upload size/row-count guardrails, undo for
  LLM-applied Tab 2 cleaning steps (#13), session/file/event audit logging to
  SQLite, a Memory Browser (Tab 3) over the ChromaDB-backed suggestion store.

## Next (open issues, scoped and ready to pick up)

- **`sanitize_for_csv()` crash** (#25) — `DataFrame.applymap` was removed in
  current pandas; the CSV-export formula-injection guard needs to move to
  `DataFrame.map` (or an equivalent per-column apply) or every cleaned-CSV
  download is broken.
- **Pipeline/batch quality-score feedback** (#31) — wire the already-built
  `generate_quality_report` into Tab 1's "Run Cleaning Pipeline" result and
  the batch-clean summary table, so cleaning quality is visible where people
  actually run it, not just in Tab 2.
- **Recipe step/param validation** (#33) — catch unknown step names and
  out-of-range parameters in an uploaded recipe before `run_pipeline` runs,
  with a clear per-step message instead of a silent skip or a buried
  sklearn/pandas error.
- **Headless recipe CLI** (#32) — a thin `cli.py` around the already
  UI-independent `run_pipeline`, so a saved recipe can run in cron/CI without
  Streamlit.
- **Good-first-issue polish** — step-widget parameter labels showing raw
  config keys instead of readable text (#26), and the "Dataset Multiplier"
  option description being a copy-pasted sampling-strategy blurb (#27).

## Later (direction, not yet scoped as issues)

- **True OS-level sandboxing for LLM-generated code** — `safe_exec.py`'s own
  docstring already flags this: the current subprocess isolation is strong
  defense-in-depth, but a compromised "allowed" import could still reach the
  host container. A gVisor/Docker/nsjail-style sandbox with its own network
  namespace and read-only filesystem is the real long-term fix, blocked today
  mainly by free-tier deployment targets (e.g. Render) not supporting nested
  containers.
- **Chunked/streaming CSV handling** — the 25MB / 500k-row caps in `app.py`
  are a blunt guardrail against memory exhaustion, not a real answer for
  larger files. Chunked reads or a background-job model would be needed to
  raise those limits safely.
- **Shared/multi-user memory store semantics** — the ChromaDB suggestion
  store is now session-scoped (fixed from the original cross-user leak, #9),
  but there's no way yet for a user to curate, promote, or share a cleaning
  suggestion across their own sessions deliberately — it's all implicit RAG
  retrieval today.
- **Replacing the free-tier Render deployment constraints** — several "Later"
  items (sandboxing, upload limits) are shaped by what the current free-tier
  host allows; moving to a deployment target with real container isolation
  would unblock more than one of them at once.
