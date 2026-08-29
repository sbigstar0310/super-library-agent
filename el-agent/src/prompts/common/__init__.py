"""Cross-benchmark prompt fragments, reused byte-equal across benchmark
packages (webgen, paperbench, …).

  - WORKSPACE_BLOCK              (workspace.py)     — submission/ discipline
  - EXTRACT_SYSTEM_PROMPT        (extract.py)       — cross-task extractor sys msg
  - EXTRACT_MAP_INSTRUCTION      (extract.py)       — extract_map.md writing protocol
  - LOCAL_EXTRACT_SYSTEM_PROMPT  (local_extract.py) — intra-codebase dedup sys msg
  - APPLY_SYSTEM_PROMPT          (apply.py)         — per-app library-apply sys msg
  - APPLY_USER_TEMPLATE          (apply.py)         — apply user-prompt format string

Benchmark-specific bits (LIBRARY_BLOCK, format_*_body, APP_RULES,
LIB_RULES, coding/library system prompts) live in each benchmark package.
"""

from prompts.common.apply import APPLY_SYSTEM_PROMPT, APPLY_USER_TEMPLATE
from prompts.common.extract import EXTRACT_MAP_INSTRUCTION, EXTRACT_SYSTEM_PROMPT
from prompts.common.local_extract import LOCAL_EXTRACT_SYSTEM_PROMPT
from prompts.common.workspace import WORKSPACE_BLOCK


__all__ = [
    "APPLY_SYSTEM_PROMPT",
    "APPLY_USER_TEMPLATE",
    "EXTRACT_MAP_INSTRUCTION",
    "EXTRACT_SYSTEM_PROMPT",
    "LOCAL_EXTRACT_SYSTEM_PROMPT",
    "WORKSPACE_BLOCK",
]
