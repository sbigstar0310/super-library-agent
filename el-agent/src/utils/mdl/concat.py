"""Legacy concat-mode MDL scoring (pre-2026 papers).

Definition::

    MDL_concat  =  NLL(library)
                 + Σ_n  NLL( app_n | library_prefix )

where every app file is scored conditioned on the **entire** library text as
a single prefix. This is the original MDL formulation from earlier
internal reports; the current measurement standard is dep-aware (see
``scoring.py``). Concat is preserved here only for cross-paper sanity
checks against pre-2026 numbers.

Known caveats (not fixed in this module — by design):

- Prefix-skip token boundary: ``library_tokens`` is computed once from the
  *stripped* library; the same stripped library is then used as the
  concat prefix. The skip is consistent within this module, but if a
  caller passes ``library_code`` that doesn't match the precomputed
  tokens, the per-file NLL split will be off. The CLI path here always
  re-reads and re-scores from ``lib_dir``, so this is safe in practice.
- 1-level dep awareness is *not* applied — that's the whole point of concat.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .configs import TaskConfig, load_task_config
from .file_io import read_app_code_files, read_dir_to_text
from .scoring import AppMDLResult, FileDetail, MDLMetric


async def score_concat(
    metric: MDLMetric,
    app_dir: str,
    lib_dir: str | None = None,
    *,
    task: TaskConfig | None = None,
    strip_comments: bool = True,
    precomputed_lib: tuple[float, int] | None = None,
) -> AppMDLResult:
    """Score one app with concat-mode MDL. See module docstring for definition.

    ``precomputed_lib`` works the same as in ``MDLMetric.ascore`` — pass
    ``(library_nll, library_tokens)`` to skip re-scoring the lib.
    """
    if task is None:
        task = load_task_config("webgen")

    parser = task.parser_module
    strip_fn = parser.strip_comments if strip_comments else None

    # Read + strip the library once (per-file strip before markdown wrap;
    # post-wrap strip is broken for JS — see file_io.py docstring).
    library_code = read_dir_to_text(lib_dir, task=task, strip_fn=strip_fn) if lib_dir else ""

    if precomputed_lib is not None:
        library_nll, library_tokens = precomputed_lib
    else:
        if library_code:
            lib_lp, lib_lt = await metric.get_prompt_logprobs(library_code)
            library_tokens = len(lib_lt)
            library_nll = -1.0 * float(np.sum(lib_lp))
        else:
            library_nll, library_tokens = 0.0, 0

    app_files = read_app_code_files(app_dir, strip_fn, task=task)
    if not app_files:
        raise ValueError(f"No app code files found under {app_dir}")

    total_app_nll = 0.0
    total_app_tokens = 0
    file_details: list[FileDetail] = []

    for rel_path, content in app_files:
        concat_text = metric._concat(library_code, content)
        concat_lp, concat_lt = await metric.get_prompt_logprobs(concat_text)
        # Skip the library_tokens-worth of prefix → remaining logprobs belong
        # to the app file. ``min(...)`` guards against off-by-one at boundary.
        prefix = min(library_tokens, len(concat_lt), len(concat_lp))
        app_lp = concat_lp[prefix:]
        file_tokens = max(0, len(concat_lt) - prefix)
        file_nll = -1.0 * float(np.sum(app_lp))
        npt = file_nll / file_tokens if file_tokens > 0 else float("nan")

        file_details.append(FileDetail(
            rel_path=rel_path,
            nll=file_nll,
            tokens=file_tokens,
            nll_per_token=npt,
        ))
        total_app_nll += file_nll
        total_app_tokens += file_tokens

    return metric._build_result(
        os.path.abspath(app_dir),
        total_app_nll, total_app_tokens,
        library_nll, library_tokens, file_details,
    )
