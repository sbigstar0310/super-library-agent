# cc-sandbox: Isolated execution environment for WebGen-Bench cc experiments.
#
# Each coding/extract/apply sub-agent runs in its own container so that:
#   - CLAUDE.md and MEMORY.md are never injected (--bare + no host mounts)
#   - Sibling task files are invisible (only workspace dir mounted)
#   - Previous experiment backups are invisible
#   - Claude Code version is pinned for reproducibility
#
# The host mounts only:
#   - The agent's own workspace directory (read-write)
#   - Reference dirs (library, baseline submissions) as read-only
#   - ~/.claude/.credentials.json read-only for OAuth auth
#
# Node.js is required because the agent may run `npm install / npm run build`
# to verify its own output.

FROM node:20-slim

# Pin Claude Code CLI version. Bump intentionally — not during experiments.
ARG CLAUDE_CODE_VERSION=2.1.140
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Writable HOME for arbitrary --user uid (host UID may not match any user in
# the image). /sandbox/.claude must pre-exist as a directory so the bind mount
# for .credentials.json lands at the right path.
# Non-root is required: Claude Code refuses --dangerously-skip-permissions
# when running as uid 0.
RUN mkdir -p /sandbox/.claude && chmod -R 777 /sandbox
ENV HOME=/sandbox

WORKDIR /workspace
CMD ["bash"]
