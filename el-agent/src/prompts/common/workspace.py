"""WORKSPACE_BLOCK — submission-target reminder for the user prompt
(coding / apply / local_extract phases).

Byte-equal across webgen and paperbench.
"""

from __future__ import annotations


__all__ = ["WORKSPACE_BLOCK"]


WORKSPACE_BLOCK = """
[Workspace]
- Your submission target is `{workspace_dir}`. Every file you write MUST live
  under that directory. Files outside it are NOT preserved when this task ends.
"""
