# Review rules — ActuCore (`phanthymotus/actucore`)

Authoritative reference: **`actucore/README.md`** — it holds the card contract.

ActuCore is Perception's mirror on the execution side: Perception turns raw
streams into semantics, ActuCore turns intent into motion commands. Execution
models (VLA policies, navigation, grasp policies, locomotion, whole-body
control) attach as cards.

**It currently ships zero cards.** The first PRs here will either add a card or
change the host. Judge them differently: a host change affects every future
card, a card change is self-contained.

## The card contract

Cards are duck-typed — no base class, no ABC, no registry decorator. Required
members: `PREFIX`, `__init__(cfg, executor)`, `get_tools()`, `dispatch(name, args)`.

Four failure modes worth checking on any card PR:

- **`PREFIX` containing an underscore.** `ActuCoreBundle.dispatch()` routes with
  `full_name.partition("_")`, so `PREFIX = "grasp_policy"` can never be reached.
  The symptom is a tool that lists fine and always answers "Unknown tool".
- **`inputSchema.properties.action.enum` missing `"info"`.** Agent Core probes
  liveness by calling the tool with `{"action": "info"}`; without it the card
  registers and stays permanently offline.
- **`dispatch()` returning a pre-wrapped `[{"type": "text", ...}]`.** It must
  return a plain dict — the MCP HTTP handler does the JSON-RPC wrapping, so
  pre-wrapping double-encodes and breaks the dashboard's parsing.
- **`x-completion` / `x-hooks` placed at the tool's top level.** They belong
  inside `inputSchema`.

Also check that a new card is actually registered: writing `plugins/<name>.py`
does nothing until an `if` block is added to `ActuCoreBundle.__init__` and the
switch is added to `config.yaml`. Card discovery is explicit, not directory
scanning.

## The `vla` card and its providers

Architecture and the reasoning behind each rule: `docs/vla-integration.md`. The
card is one card with pluggable providers — **not one card per model** — and the
two axes (which model, which robot) are deliberately orthogonal.

**A new provider** is a file in `plugins/vla/providers/` exposing
`PROVIDER(descriptor, config=None, on_status=None)` plus `capabilities` / `infer`
/ `health` / `close`. Check:

- **`capabilities()` must be answerable before the weights are loaded.** That is
  what lets negotiation refuse a mismatched action space before several GB land on
  disk. A provider that reads its config in `__init__` and loads weights in a
  background thread is the shape to look for; one that blocks the constructor on a
  model load is not.
- **Claimed capabilities must be implemented.** `supports_rtc` is the live example:
  LeRobot ships RTC for the flow-matching policies, but the `smolvla` provider does
  not implement the prefix conditioning it needs, so it reports `False`. A provider
  claiming a capability it lacks makes the card hand it an `inference_delay`
  nothing acts on.
- **Lazy import.** torch / lerobot imported inside the functions that need them —
  a card nobody selected must not pull a GPU stack into the process, and on the
  jp5.11 line those libraries are permanently absent.
- **`on_status` accepted even when the provider downloads nothing.** The card
  passes it without asking which provider it built. See §"Model downloads" above
  for what it is for.
- **Weights pinned and progress-reporting** — same two rules as every other model
  in this project.

**Negotiation belongs at `start`, not per command.** A new capability or descriptor
field that needs checking goes in `negotiate.check()`, which collects *every*
disagreement and names the two numbers ("模型输出 32 维动作，下游只接受 14 维"). The
alternative is discovering the mismatch one command at a time at 30 Hz, by which
point the arm has moved.

**Do not default safety-relevant fields.** `message.build()` requires `ttl_ms`,
`stamp_ms` and `obs_stamp_ms` explicitly, because a default invented at the sender
only moves the mistake somewhere harder to see. `stamp_ms` and `obs_stamp_ms` are
different numbers — a PR that passes the same value for both has disabled the
receiver's staleness check while every latency-free test still passes.

## Tool `type` drives scheduling

`type` is one of `sensor` / `actuator` / `processor` / `resource`, and it changes
how Agent Core dispatches: consecutive `sensor` calls batch in parallel, while
`actuator` and `processor` pass the ACP barrier and wait for pending actions
first (`agent-core/src/event/llm.py::_needs_barrier`). An undeclared `type`
defaults to barrier-guarded, which is the safe side.

Execution models are normally `processor`. A card declaring `sensor` for
something that moves the robot is a real bug — it would skip the barrier.

Long-running actions (navigate to a point, execute a grasp) should declare
`x-completion` so the barrier knows when they finish. A blocking `dispatch()`
that holds the HTTP request for the duration of a motion is the wrong shape.

## Jetson only

There is exactly one Dockerfile — `Dockerfile.jetson`. Execution models need the
GPU, so `deploy/build_actucore.sh` takes no `--variant`, only `--jp-version`
(5.11 / 6.1). Build context is the **repo root**, not `actucore/`, because the
image also needs `deploy/ros-base/audio_msgs/` — so `COPY` paths inside it are
`actucore/…`. A PR adding a `COPY` with a path relative to `actucore/` will fail
the build.

The image is deliberately thin: base CUDA torch + ROS2 + `pyyaml requests`,
nothing else. A card's dependencies belong in their own `RUN` layer. Watch for
PRs that pile a card's heavyweight deps into the shared base layers — that is
how perception's image reached several GB.

The image also copies two files out of `perception/utils/` — `model_downloader.py`
and `model_progress.py` — flat into `/work`, so cards import them by bare name.
**They travel together.** `plugins/vla/providers/smolvla.py` imports both; shipping
one without the other fails at card start, on a robot, with an `ImportError` that
no test here would have caught. A PR that adds a card fetching weights must check
both `COPY` lines are present.

## Model downloads

A card that fetches weights follows the same two rules Perception does. They are
written out in **`perception/README.md` §"Model downloads: the two rules"** — which
is *not* loaded for an actucore-only PR, so the checks are restated here rather
than cross-referenced:

1. **Pinned, and free to choose a source.** Every file carries `size` + `sha256`,
   verified before acceptance; `base_url` may be a list of hosts, probed once and
   used fastest-first. Flag a new manifest entry without pins, and a hand-rolled
   loop over sources instead of passing the list to `ensure_verified_bundle`.
2. **It must say how far along it is.** A download with no progress is
   indistinguishable from a hang. Flag an `ensure_*` call that omits `progress_cb`
   where the caller has a status channel, a status line formatted by hand instead
   of via `model_progress.fetch_status`, and an archive path with no `stage_cb`.

ActuCore does not get an exemption for reusing Perception's downloader — it is the
*caller* side these rules are about.

The stake is higher here than in Perception: these are the largest downloads in the
system (a SmolVLA checkpoint ~900 MB, its backbone another ~1 GB), and the fetch
happens inside provider construction, which happens inside `start`. So the two
things to check on any such PR:

- `ensure_verified_bundle` is called **with** `progress_cb` (from
  `model_progress.fetch_status`), not without;
- the card can answer `info` with that line while the download is in flight. The
  VLA card could not, until `_starting` was added: `_running` is set only after
  construction returns, so `_state()` reported `idle` for the whole of a
  multi-gigabyte fetch and the card looked stopped while doing the longest thing
  it ever does.

A provider factory takes `(descriptor, config, on_status=None)`. A new provider
must accept `on_status` even when it downloads nothing — the card hands it to
whichever provider it built without asking which one that is.

`COPY actucore/deploy/ /deploy/` must stay: Agent Core extracts
`/deploy/service.yml` from the image to merge the compose fragment, and dropping
it silently degrades to the legacy `docker run` path — which does **not** carry
the fragment's volumes, so the `dds-local.xml` mount disappears with it and the
container is no longer isolated. Perception's Jetson Dockerfile shipped exactly
this bug until `a916a02` ("include deployment service fragment", 2026-08-20); it
has the `COPY` now, and R1's running container does have the profile mounted.

## Ports

MCP HTTP on **15730**; 15731 is reserved but unused (unlike perception, ActuCore
has no WebSocket server — there is no audio-stream equivalent). A PR that adds a
second listener should say why.

Changing the port means changing all of: `actucore/config.yaml`, the `EXPOSE` in
`Dockerfile.jetson`, `_SERVICE_ENDPOINTS['actucore']` in
`agent-core/src/api/drivers.py`, the register payload in
`deploy/build_actucore.sh`, and the README port tables. Flag any partial move.

## Identity strings that must stay unique

`serverInfo.name` is `actucore-bundle`, and the registration name is `ActuCore`
with `category: "actucore"`. Agent Core dedupes MCP entries by url / name /
server_name, so a name collision with perception makes the two evict each other.
The `registryImage` (`actucore`) must also keep matching the `_SERVICE_ENDPOINTS`
key, or the deploy manifest loses its port and mcp_url.
