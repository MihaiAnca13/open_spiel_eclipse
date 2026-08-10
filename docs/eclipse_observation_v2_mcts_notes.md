# Deferred MCTS notes for Observation V2

If play-time search is revisited after V2, reuse `async_mcts.MCTSBot` with
prior-aware PUCT selection and actor-only rollouts; do not use the unvalidated
critic as a leaf value. Before any raw-state rollout or UI wiring, determinize
face-down discovery identities and bag order from the public V2 ledger at each
search root. Keep UI integration deferred until the V2 observation and its
grounding tests are complete.
