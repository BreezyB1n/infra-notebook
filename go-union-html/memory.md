# go-union-html automation memory

## 2026-05-30

- Topic: `v2/adfetch/build_context/graph`
- Output: `2026-05-30-build-context-graph.html`
- Why chosen: module is compact, central to `BuildFlowContext`, and good for explaining DAG scheduling, dependency triggering, and timeout fallback.
- Key anchors:
  - `v2/adfetch/build_context/build_context.go`
  - `v2/adfetch/build_context/graph/interface.go`
  - `v2/adfetch/build_context/graph/dag.go`
  - `v2/adfetch/build_context/graph/dag_plan_debug.go`
  - `v2/adfetch/build_context/graph/dag_test.go`
- Run summary: wrote a standalone HTML lesson focused on DAG scheduling, dependency counters, timeout recovery, and the real `BuildFlowContext` integration point. Recorded at 2026-05-30 12:05:27 CST.
