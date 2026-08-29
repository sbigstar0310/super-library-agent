"""WebGen-Bench mswe agents — package marker.

Five peers:
  - WebgenCodingAgent          baseline / sla_naive / sla_ours coding pass
  - WebgenApplyAgent           sla_ours per-app refactor against lib
  - WebgenGlobalExtractAgent   sla_ours cross-app extract → ui-lib
  - WebgenLocalExtractAgent    sla_ours per-app intra-extract → src/local_lib
  - WebgenLibraryAgent         sla_naive unified extract + apply (single-shot)
"""

from mswe_agents.webgen.apply_agent import WebgenApplyAgent
from mswe_agents.webgen.coding_agent import WebgenCodingAgent
from mswe_agents.webgen.global_extract_agent import WebgenGlobalExtractAgent
from mswe_agents.webgen.library_agent import WebgenLibraryAgent
from mswe_agents.webgen.local_extract_agent import WebgenLocalExtractAgent

__all__ = [
    "WebgenApplyAgent",
    "WebgenCodingAgent",
    "WebgenGlobalExtractAgent",
    "WebgenLibraryAgent",
    "WebgenLocalExtractAgent",
]
