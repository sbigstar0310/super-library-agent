# WebGen layout specs

One `<task_id>.md` per task, describing the page-by-page layout of a reference implementation:
header, main regions, forms, interactive elements, position vocabulary. 62 of the 101 WebGen-Bench
tasks have one, covering all 24 tasks of the reported suites (clusters 2, 5, 13) and all 16 of
`diverse16`.

`WebgenCodingAgent` looks up `<task_id>.md` here and appends it to the task under a
`[Visual & functional layout reference]` heading; a task with no file gets its original instruction
unchanged. `LAYOUT_SPECS_DIR` points here by default, and `LAYOUT_SPECS_DIR=` (empty) turns the
augmentation off for a run.

They exist to remove a confound. A WebGen instruction is one line and leaves the page count, the
shape of the login flow and the form fields open. An agent that can also see a shared library
resolves that freedom differently: it grows the scope of the app to use what the library offers. The
arm with a library then writes a bigger app than the arm without one for reasons unrelated to reuse.
Fixing the visible spec across arms keeps the comparison on code structure.

A spec is one task and one reference implementation; cluster-shared specs are not supported. To add
or regenerate one, see `scripts/layout_specs/README.md`. Overwrite a spec in place rather than
keeping `.v1.md` / `.v2.md` beside it, since git already has the old one.
