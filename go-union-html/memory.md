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
- Publish summary: synced automation files into `/Users/bytedance/dev/infra/infra-notebook/go-union-html/`, committed only that directory on `main`, and pushed commit `dbfcbe0` to `origin/main`.
- Publish runtime: ~00:04

## 2026-06-01

- Topic: `v2/adfetch/flow_check`
- Output: `2026-06-01-flow-check.html`
- Why chosen: fresh topic after the DAG lesson; it sits exactly between `protocol_mapping` and `build_flow_context`, has a clear boundary, and is ideal for teaching the "request admission gate" concept.
- Key anchors:
  - `v2/adfetch/ad_fetch.go`
  - `v2/adfetch/flow_check/flow_check.go`
  - `v2/adfetch/flow_check/checker_interface.go`
  - `v2/adfetch/flow_check/base_check.go`
  - `v2/adfetch/flow_check/sdk_check.go`
  - `v2/adfetch/flow_check/media_config.go`
  - `v2/adfetch/entity/bid_request_with_rit_info.go`
  - `v2/adfetch/flow_check/sdk_check_test.go`
- Run summary: wrote a standalone HTML lesson explaining `flow_check` as the pre-engine admission gate, covering config binding, checker composition, SDK vs DSP rule separation, and a concrete dynamic-layout block example. Recorded at 2026-06-01 09:47:36 CST.
- Publish summary: synced automation files into `/Users/bytedance/dev/infra/infra-notebook/go-union-html/`; pending git commit and push details.
- Next topic hints:
  - `v2/adfetch/protocol_mapping/sdk` for the step right before `flow_check`
  - `v2/adfetch/context_drop` for the next "why was the request still stopped" layer
  - `v2/adfetch/request_log` for how this stage gets observed afterward
- Current run note: created the HTML lesson, refreshed automation memory, and synced the lesson files into `infra-notebook`. Runtime so far: about 00:09 before git publish.
