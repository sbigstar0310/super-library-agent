"""ResumeAgent — DefaultAgent variant that can resume a prior message history.

When ``run()`` gets ``prior_messages``, they are sanitized (L6 strip) and used
as the initial history and the new task becomes a follow-up user message,
rather than replacing the system+user templates. Without ``prior_messages``,
behavior is identical to ``DefaultAgent.run()``.
"""

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow


def sanitize_messages_l6(messages: list[dict]) -> list[dict]:
    """Drop fields that don't roundtrip cleanly when replayed to the model.

    Strips agent-side bookkeeping ('extra' dicts, template vars) that the API
    rejects on resume, keeping message identity intact. Also drops any trailing
    assistant message with orphan ``tool_calls``: DefaultAgent's terminal step
    is a final tool_call followed only by an internal 'exit' marker, so once we
    strip that marker the tool_call is orphaned and strict-mode providers
    (DeepSeek) reject the next request ("'tool_calls' must be followed by tool
    messages"). Removing it is safe — resume picks up from the prior tool
    result and the model decides afresh after the appended feedback.
    """
    sanitized: list[dict] = []
    drop_keys = {"extra", "template_vars"}
    for m in messages:
        if not isinstance(m, dict):
            continue
        clean = {k: v for k, v in m.items() if k not in drop_keys}
        # The 'exit' marker is DefaultAgent-internal; replaying it would
        # short-circuit the resume loop before the new task is processed.
        if clean.get("role") == "exit":
            continue
        sanitized.append(clean)

    # Strip trailing assistant messages whose tool_calls have no following
    # tool responses (orphan due to 'exit' marker removal above).
    while sanitized:
        last = sanitized[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            sanitized.pop()
            continue
        break
    return sanitized


class ResumeAgent(DefaultAgent):
    """DefaultAgent + optional resume-from-prior-messages flow."""

    def run(
        self,
        task: str = "",
        *,
        prior_messages: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        if not prior_messages:
            return super().run(task=task, **kwargs)

        # Resume path: preserve history, append new task as a user message.
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = sanitize_messages_l6(prior_messages)
        if task:
            self.add_messages(
                self.model.format_message(role="user", content=task)
            )

        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages and self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})
