# Synthetic workflow validation

Hypothesis: the shared runner can fit, save/reload and transfer features between
chronological prefixes, enforce promotion gates and replay a final export.
Comparator: a fixed, explicitly synthetic constant ensemble with the five V6
member roles. This measures workflow behavior, not an improvement over V6.
Stopping rule: any failed gate or validation error ends the run; no retries.

See [the complete commands and contracts](../../docs/ML_WORKFLOW.md). All neural
fits use two tiny synthetic epochs; CatBoost fits 24 trees. The recipe updates
both dim8 and dim16 fixture members and remains incapable of research output.
