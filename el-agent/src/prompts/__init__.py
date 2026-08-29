# Package marker for `prompts`. Domain prompts live in subpackages
# (`prompts.common`, `prompts.paperbench`, `prompts.webgen`, ...). The old
# top-level re-exports (`feedback_prompts`, `function_test_prompt`) were
# removed in the archive cleanup and had no remaining importers; keeping this
# init empty avoids crashing every `from prompts.<subpkg> import ...`.
