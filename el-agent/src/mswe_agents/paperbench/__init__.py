"""PaperBench mswe agents — package marker.

Five peers (mirrors `mswe_agents/webgen/`):
  - PaperbenchCodingAgent          baseline / sla_naive / sla_ours coding pass
  - PaperbenchApplyAgent           sla_ours per-paper refactor against shared lib
  - PaperbenchGlobalExtractAgent   sla_ours cross-paper extract → lib/
  - PaperbenchLocalExtractAgent    sla_ours per-paper intra-extract
  - PaperbenchLibraryAgent         sla_naive unified extract + apply (single-shot)

Entry point: `run/paperbench_full_run.py` (round-based orchestrator).
"""

from mswe_agents.paperbench.apply_agent import PaperbenchApplyAgent
from mswe_agents.paperbench.coding_agent import PaperbenchCodingAgent
from mswe_agents.paperbench.global_extract_agent import PaperbenchGlobalExtractAgent
from mswe_agents.paperbench.library_agent import PaperbenchLibraryAgent
from mswe_agents.paperbench.local_extract_agent import PaperbenchLocalExtractAgent


__all__ = [
    "PaperbenchApplyAgent",
    "PaperbenchCodingAgent",
    "PaperbenchGlobalExtractAgent",
    "PaperbenchLibraryAgent",
    "PaperbenchLocalExtractAgent",
]
