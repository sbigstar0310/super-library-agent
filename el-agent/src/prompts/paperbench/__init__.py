"""PaperBench mswe-agent prompt package.

Six modules:

  - paperbench.py          APP_RULES + LIB_RULES (Python codebase)
  - common.py              WORKSPACE_BLOCK + LIBRARY_BLOCK + format_paper_body
  - coding_agent.py        upstream-verbatim system + user prompt
                           (paperbench basicagent reference parity)
  - local_extract_agent.py intra-paper extract — agent autonomous placement
  - global_extract_agent.py cross-paper extract + EXTRACT_MAP_INSTRUCTION
  - apply_agent.py         per-paper refactor against the shared lib
  - library_agent.py       sla_naive single-shot extract + apply
"""
