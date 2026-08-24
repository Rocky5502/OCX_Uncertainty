# Automated Reviewer Prompt Template

The campaign instantiated this template with a frozen task, repository context,
retrieved evidence, starting implementation, and condition-specific display.
Task-bearing prompts are not redistributed because they contain third-party
repository source; their SHA-256 hashes remain in `protocol_freeze.json`.

```text
# Response contract
Review the starting implementation using only the supplied material. Do not
browse the web, run code, use external tools, infer hidden tests, request more
context, or describe repository files that are not shown.

Return exactly this plain-text structure, with no text outside it:

STARTING_CORRECT: true|false
STARTING_CONFIDENCE: integer 0-100
FINAL_CONFIDENCE: integer 0-100
FINAL_CODE_BEGIN
<exactly one complete target function; no imports or surrounding declarations>
FINAL_CODE_END

# Repository review task
{{TASK_TEXT}}

# Repository context
{{REPOSITORY_CONTEXT}}

# Retrieved evidence
{{RETRIEVED_EVIDENCE}}

# Starting implementation
{{STARTING_CODE}}

{{UNCERTAINTY_TRACE_FOR_DISPLAY_AND_TARGETED_CONDITIONS}}
{{RECOMMENDED_ACTION_FOR_TARGETED_CONDITION}}
```

`STARTING_CONFIDENCE` is the probability that the starting-correctness judgment
is right. `FINAL_CONFIDENCE` is the probability that the returned function
passes the withheld executable tests. The response always contains one complete
target function, even when no edit is needed.
