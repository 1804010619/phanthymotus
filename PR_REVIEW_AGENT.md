# PR Review Agent

Automates the manual PR loop: enter the PR branch, work out what to build,
build it, report the image tag or the build error, and write a review.

Trigger it by commenting on a PR:

```
/request_bot_review
```

## Why polling, not webhooks

The default trigger is **polling**, not a GitHub webhook. Polling needs only
outbound network access, so the agent runs behind NAT with no public IP, no
open port, no TLS certificate, and no webhook registration.

Once per interval, per repo, it makes a single API call:

```
GET /repos/{owner}/{repo}/issues/comments?since=<watermark>
```

A PR is an issue underneath, so that one endpoint returns PR conversation
comments too. At the 30s default across two repos that is ~240 requests/hour
against a 5000/hour quota.

The cost is up to one interval of latency. For a flow whose next step is a
10–20 minute Docker build, that is not noticeable.

The webhook endpoint (`POST /webhook`) still exists and can be enabled if the
host ever becomes reachable — both paths share the same trigger and dedup
logic, so they can run simultaneously without double-triggering.

## Commands

| Command | Effect |
|---------|--------|
| `/request_bot_review` | Detect build targets, build, then review |
| `/request_bot_review skip-build` | Review only |
| `/request_bot_review build-only` | Build only |
| `/request_bot_review force` | Re-review a commit that was already reviewed |
| `/request_bot_review core` | Force the `core` target |
| `/request_bot_review perception` | Force the `perception` target |
| `/request_bot_review unitree/g1` | Force a specific driver |

The trigger must start a line, and it must be in the PR's **main conversation**
box. Line-level review comments are a different GitHub event and are not seen.

## Build target detection

Determined from `git diff --name-only origin/main...HEAD`.

**phanthymotus**

| Changed path | Target |
|--------------|--------|
| `agent-core/**` | `deploy/build_core.sh` |
| `perception/**` | `deploy/build_perception.sh` |
| `README*`, `docs/**`, `CODEOWNERS` | none — review only |

**phanthymotus-driver**

| Changed path | Target |
|--------------|--------|
| `{provider}/{model}/**` where that directory has `driver.yaml` + `Dockerfile` | `build.sh {provider}/{model}` |
| `{provider}/{model}/**` without those markers (e.g. `dji/base/`) | none — not a buildable driver |
| `build.sh`, `README*` | none — review only |

Driver discovery probes the worktree for `driver.yaml` + `Dockerfile`, the same
test `build.sh` applies. A hardcoded provider list would silently skip every
newly added vendor: that is exactly what happened to PR #166 adding
`robotera/q5_bundle`, which got a review and no image.

The agent invokes the repos' existing build scripts rather than reimplementing
the build. Image tags, mirrors, and registry handling stay in one place.

### How each target is deployed

The build-result comment tailors its instructions per target, because the three
are deployed differently:

| Target | Ships `deploy/service.yml` | How it is deployed |
|--------|---------------------------|--------------------|
| driver | yes, per driver | Agent Core extracts it from the image and merges it into the host compose file (`api/drivers.py:_deploy_sync`) |
| perception | yes | same path as drivers |
| core | no | updated in place through the web console: `POST /api/system/update` pulls the image and hands over to a restart-helper container |

So drivers and perception get a one-line command in the comment:

```bash
./deploy/run-pr-image.sh <image-ref>
```

`deploy/run-pr-image.sh` (committed in both repos) pulls the image, reads the
compose fragment out of `/deploy/service.yml`, substitutes the real ref, and
starts it from a standalone compose file under `PR_IMAGE_DIR`. Then `--logs`,
`--shell`, `--status`, `--down`.

Using compose rather than a generated `docker run` matters for three reasons:
it is the same tool production uses, the flags each service declares
(privileged, host networking, device mounts) are used exactly as its author
wrote them instead of being re-derived, and `--down` tears everything down
without touching the host's real compose file.

Core gets only the image reference and a pointer to the web console. It ships no
service fragment, and starting a second copy by hand would fight the running
agent — updating it requires the restart helper that `POST /api/system/update`
hands over to.

The script needs nothing beyond docker: the fragment is wrapped with text
transforms, so there is no python or yq dependency on the machine under test.

## Parallelism and isolation

`MAX_CONCURRENT_JOBS` (default 2) workers pull from an async queue.

Isolation comes from git worktrees. Each job checks out into its own directory
under `/data/repos/worktrees/`, so two concurrent builds never share a working
tree. The only shared state is:

- **the bare clone** — fetch-only, guarded by a per-repo lock
- **the Docker daemon** — serializes internally

Raising `MAX_CONCURRENT_JOBS` is safe as long as the host has the CPU and disk
for concurrent Docker builds.

## Git strategy

One bare clone per repo, created once, then fetched incrementally:

```
/data/repos/
  phanthymotus.git/                    # git clone --bare, once
  phanthymotus-driver.git/
  worktrees/
    phanthymotus-pr-42-a1b2c3d/        # transient, removed after the job
  poller_state.json                    # watermarks + processed comment IDs
```

Each job runs `git fetch` (seconds) and `git worktree add`, then merges the PR
onto `origin/main` — mirroring what `enter_pr_branch.sh` does by hand. A merge
conflict is reported to the PR and not retried; only the author can fix it.

This volume must persist across restarts. Losing it forces full re-clones and
makes the poller replay up to `POLL_INITIAL_LOOKBACK_MINUTES` of comments.

## Timeouts and retries

Two independent timeouts:

- `BUILD_TIMEOUT_SECONDS` (default 1800) — one `docker build` invocation
- `JOB_TIMEOUT_SECONDS` (default 3600) — the whole pipeline for one attempt

A job exceeding the job timeout is presumed lost and retried, up to
`MAX_ATTEMPTS` (default 3) with `RETRY_BACKOFF_SECONDS` between attempts. When
attempts run out, the PR gets a failure comment listing each attempt's reason.

Not everything is retried. Retrying costs an hour, so it is reserved for
failures another attempt could plausibly fix:

| Outcome | Retried | Why |
|---------|---------|-----|
| Job timeout | yes | Presumed hung or lost |
| Network / git / registry error | yes | Usually transient |
| Merge conflict | no | Only the author can resolve it |
| Build failure (non-zero exit) | no | A real result — rebuilding says the same thing |

> **Sizing note.** Builds run sequentially, so a PR touching N drivers needs
> roughly N × `BUILD_TIMEOUT_SECONDS`. With the defaults, three drivers can
> exceed the 1h job timeout and be retried as "lost" even though the build was
> progressing. Raise `JOB_TIMEOUT_SECONDS` if that is common.

When a job is cancelled or times out, the build subprocess is killed by process
group — `bash`, `docker`, and `buildx` all die together. Without that, an
orphaned `docker build` would keep holding CPU and build cache while the retry
competed with it.

## PR comment lifecycle

One comment tracks the job, edited in place, so a PR does not accumulate one
comment per stage:

```
Request accepted → Building... → Build Result (image tag, or logs on failure)
```

The review is posted as a **separate** comment, so the build result stays
readable alongside it.

If a new request arrives for a PR whose earlier job is still queued, the old
one is superseded and its comment says so. A job already *running* is left to
finish — its build is expensive and its result is still valid for the commit it
started on.

## Review

Two phases: deterministic checks, then an agentic loop that explores the
checkout.

### Deterministic checks

Run first, and their results are handed to the loop as established facts so it
does not waste rounds re-deriving them.

- **File size** — every added/modified file is `stat`ed in the worktree, so text
  and binary are measured alike. Anything at or above `LARGE_FILE_THRESHOLD_KB`
  (default 500) is reported in its own **Large files** section pointing at the
  COS convention. The earlier version parsed `git diff --stat`, which reports
  bytes *only* for binary files — a 2 MB generated JSON passed silently.
- **Archive and binary extensions** — `.tar.gz`, `.zip`, `.so`, `.pt`, `.onnx`,
  `.whl` and similar, flagged regardless of size. Every real offender in these
  repos is under 1 MB (a committed `.zip`, an x86_64 `.so` in an ARM64-only
  project), so a size threshold alone misses all of them. `.gitignore` covers
  only images, so this check is the only gate.
- **Infrastructure** — every touched `Dockerfile*`, `requirements.txt`,
  `pyproject.toml`, `build*.sh`, `driver.yaml`, `deploy/service.yml`, tiered by
  blast radius:

  | Path | Who depends on it |
  |------|-------------------|
  | `phanthymotus/deploy/ros-base/Dockerfile` | all drivers + agent-core + perception, **across both repos** |
  | `phanthymotus-driver/dji/base/*` | the three DJI drones |
  | `phanthymotus-driver/common/**` | every driver that imports it |
  | one component's Dockerfile | that component |

  Shared paths count as infrastructure whatever they are named — `common/` is
  ordinary Python that every driver imports, so a filename test alone would miss
  the highest-blast-radius changes.
- **Possible secrets** — `.env`, `credentials`, `secret`, `.pem`, `.key`
  (`*.example` and `*.sample` exempt).

### The review loop

An LLM with read-only tools over the PR's checkout, bounded by
`REVIEW_MAX_ROUNDS` (20) and `REVIEW_TIMEOUT_SECONDS` (600).

| Tool | Purpose |
|------|---------|
| `list_dir(path)` | entries with type and size |
| `read_file(path, start_line, max_lines)` | line-numbered text |
| `grep(pattern, path, glob)` | matches as `file:line: text` |
| `file_diff(path)` | this PR's diff for one file |
| `finish_review(summary, issues, suggestions)` | terminal |

**The prompt no longer carries the diff.** It carries the file list, `--stat`,
and the deterministic results; the loop reads what it needs. That structurally
removes the old failure where a large PR built a prompt past the model's
context, which `MAX_DIFF_LINES` was papering over.

Modelled on agent-core's `subagent/agent.py` rather than its main `event/llm.py`
loop, for reasons the main loop demonstrates by counter-example: it has no
wall-clock timeout (500 rounds x a 120 s read timeout runs for hours), it calls
`json.loads` on tool arguments unguarded so one malformed blob kills the turn,
and it breaks silently at its round ceiling. Here the budget is bounded in both
rounds and seconds, malformed arguments degrade to `{}` so the tool can report
the missing parameter, tool failures come back as `[tool error] ...` content the
model can correct, and exhaustion is reported explicitly — **a review that was
cut short must not look like a review that found nothing**, so both the PR
comment and the dashboard say so.

`LLM_BASE_URL` accepts a bare host, a `/v1` root, or the full endpoint — `/v1`
is added when missing. A gateway that serves its web UI at `/chat/completions`
would otherwise answer 200 with HTML, and the failure would read as a JSON
parse error rather than a wrong URL.

### Rules, docs and reference implementations

`agents/pr_review/rules/*.md` hold the review standards, so changing them is
editing markdown. `common.md` always applies; `driver.md`, `core.md` and
`perception.md` are added by detected component. `components.py` maps each
component to its authoritative docs and a comparable existing implementation,
both named in the prompt.

The rules are written from what the repos actually document *and* actually do,
which diverges more than once:

- A driver's `dispatch()` must return a **plain dict**. This is the single
  highest-value check because `README_dev.md` contradicts itself — line 246 bans
  the pre-wrapped `[{"type": "text", ...}]` form while its own skeleton example
  around line 453 does exactly that. Anyone copying the example ships a
  double-encoded payload that looks like a rendering bug.
- Driver ports must be verified against the other `driver.yaml` files, **not**
  the table in `README.md`, which is already wrong. Four drivers really do
  declare 15702 and two declare 15703.
- `driver.md` also carries a **do not flag** list, so the loop does not fight
  conformant code: `README_dev.md` forbids `network_mode`/`ipc`/`pid` in
  `deploy/service.yml` but every existing driver sets them, and the doc's
  `drivers/<provider>/<model>/` paths have no `drivers/` prefix in reality.

For a new driver the reference is chosen by shape — `unitree/go1` for structure,
`robotera/q5_bundle` for decomposition, `dji/mavic3e` for a native-SDK bridge,
`pnpbotics/adam` for gRPC, `unitree/go2` for SLAM. `deep_robotics/lynx_m20` and
`unitree/g1/device.py` are deliberately excluded as models.

### Sandbox

The loop reads a worktree built from an **untrusted PR**, so both file names and
file contents are attacker-authored, and the review is posted to a **public** PR
comment. Every path is resolved and checked against the worktree root. Because
`Path.resolve()` follows symlinks, that also blocks the dangerous case: a PR
adding `evil -> /proc/self/environ` (which holds `GITHUB_TOKEN`,
`REGISTRY_PASSWORD` and `LLM_API_KEY`) and getting the agent to read it into
public. Confinement also keeps `jobs.db`, `poller_state.json` and the bare
clones out of reach, since they live one level up in `$DATA_DIR`. An absolute
path is reinterpreted as repo-relative rather than refused, so it reads nothing
outside. `.git/` is excluded, binaries are refused rather than returned as
bytes, and every result is capped so one `read_file` cannot blow the context.

**There is no shell or exec tool**, despite `ls`/`grep`/`cat`/`diff` being the
requested capabilities — they are provided as fixed, argument-validated,
read-only tools instead. Adding `exec` would add an execution path and no review
capability.

What this does *not* address, stated plainly: the agent runs `docker build` on
Dockerfiles from untrusted PRs, so a malicious `RUN` executes on the build host.
That is inherent to "build the PR" and independent of the review loop; it is
where to look first if this ever needs hardening.

## Dashboard

A web UI on the same port shows live status, review history, and full build logs.
Vanilla ES modules, no build step, reusing agent-core's design tokens.

Open it directly:

```
http://<host>:25000/
```

`BIND_ADDR` (in `.env`) controls who can reach it. It defaults to `0.0.0.0`,
published by compose as `${BIND_ADDR}:${PORT}:${PORT}`. To restrict it to
loopback and tunnel in instead:

```bash
# .env
BIND_ADDR=127.0.0.1
```
```bash
ssh -L 25000:localhost:25000 <user>@<host>
# then open http://localhost:25000
```

Note that `HOST` and `BIND_ADDR` are different things: `HOST` is what uvicorn
binds *inside* the container (leave it at `0.0.0.0`), while `BIND_ADDR` is what
compose publishes on the host.

Three views:

- **Overview** — queue depth, in-flight jobs, poller health (last poll, last
  error), effective config. Polls every 5s.
- **History** — every job, filterable by status and repo, paged. Survives
  restarts.
- **Job detail** — metadata, build results with copyable image tags, a log
  viewer per build target, the rendered review, rule findings, and per-attempt
  failure reasons. Deep-linkable via `#job/<id>`.

While a build runs, its log pane tails live: the client re-requests from the
byte offset it last received. The pane only autoscrolls if you are already at the
bottom, so tailing does not yank the view away while you read an error further
up.

### Persistence

State lives on the host at `DATA_HOST_DIR` (default
`/opt/phanthy-motus/pr-review`), bind-mounted to `/data/repos` in the container:

```
/opt/phanthy-motus/pr-review/
  jobs.db              review history (SQLite)
  logs/<job>/<n>-<target>.log    full build logs
  <repo>.git/          bare clones
  poller_state.json    poller watermarks + processed comment IDs
  worktrees/           transient, removed after each job
```

Metadata in the database, bulky payloads on disk — the same split agent-core
uses for LLM request logs. A bind mount rather than a named volume, matching
agent-core's `/opt/phanthy-motus/data`: the path is discoverable, a backup is
`tar czf backup.tar.gz /opt/phanthy-motus/pr-review`, and `down -v` cannot take
the history with it.

`JOB_HISTORY_DAYS` (default 30) bounds retention. Pruning runs at startup and
deletes log directories along with their job rows, so the two never drift.

Nothing resumes across a restart, so any job left non-terminal by an unclean
shutdown (`docker kill`, OOM) is reconciled to `cancelled` at boot. A graceful
`./deploy.sh stop` already notifies those jobs on their PRs; this covers the case
that bypasses it, so the dashboard never shows work that no longer exists.

### Repeat triggers

Handled per commit, not per PR:

| Situation | Behaviour |
|-----------|-----------|
| New commit | New review — this is the normal fix-and-retrigger flow |
| Same commit, in flight | Skipped, with a comment saying so |
| Same commit, already reviewed (`review_done` / `build_failed`) | Skipped, pointing at the earlier result |
| Same commit, previous attempt produced no result (`cancelled` / `timeout` / `error`) | Allowed — those delivered nothing |
| `/request_bot_review force` | Re-reviewed regardless |

The distinction in the last two rows matters: a job killed by a restart or an
infrastructure failure must not leave a commit permanently un-reviewable. The
check reads SQLite, not the in-memory queue, which is empty after a restart and
would otherwise let a completed review be silently redone.

### API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Queue depth, active jobs, poller health, config |
| `GET /api/jobs?limit&offset&status&repo` | Paginated history |
| `GET /api/jobs/{id}` | Full detail incl. review text and findings |
| `GET /api/jobs/{id}/log/{idx}?offset=N` | Log bytes from `offset` |

Raw JSON, not agent-core's `{code, message, data}` envelope — `deploy.sh status`
curls these directly.

Log tailing is HTTP offset polling rather than a WebSocket. The project uses WS
for its data plane, but that carries high-frequency sensor data; build logs are
low-frequency text already being written to a file, so polling avoids connection
lifecycle, fan-out, and reconnect logic, and recovering from a dropped request is
just repeating it.

### Security note

**There is no authentication.** With the default `BIND_ADDR=0.0.0.0`, anyone who
can reach the host on `PORT` can read every build log, review, and the config
block from `/api/status`. Build output can contain sensitive detail.

That is an acceptable trade on a trusted private network, and it matches how the
other services on these hosts are exposed. It is not acceptable on a shared or
internet-facing host — there, set `BIND_ADDR=127.0.0.1` and tunnel.

If the dashboard ever needs real exposure, agent-core's `auth.py` is the pattern
to copy: a bearer token read from `.env`, with middleware guarding `/api/*` and
leaving the static assets open.

Everything the dashboard renders is escaped, because most of it is influenced by
whoever opened the PR: branch names, build output, error text, and LLM review
that quotes the diff. Log and error text render via `textContent`; the review's
markdown subset escapes *before* applying its patterns, which is what makes it
safe.

## Deploy

```bash
cd phanthymotus/deploy/pr-review
cp .env.example .env
$EDITOR .env          # GITHUB_TOKEN, REGISTRY_*, LLM_*
./deploy.sh
```

### Lifecycle commands

| Command | Effect |
|---------|--------|
| `./deploy.sh` / `up` | Build and start |
| `./deploy.sh rebuild` | Rebuild without cache, recreate container |
| `./deploy.sh stop` | Stop, keeping container and data |
| `./deploy.sh start` | Start a stopped container (no rebuild) |
| `./deploy.sh restart` | Restart (no rebuild) |
| `./deploy.sh down` | Remove container, keep the data volume |
| `./deploy.sh down --purge` | Also delete the data volume (prompts first) |
| `./deploy.sh status` | Container state plus the agent's `/status` |
| `./deploy.sh logs [-n N]` | Follow logs |

`stop` and `restart` are graceful. Compose allows `stop_grace_period` (30s),
during which the agent posts an interruption notice on the PR of every job that
was queued or in flight — otherwise a build killed mid-run would leave a comment
frozen at "Building..." forever. The grace window is deliberately too short to
finish a build: waiting out a 30-minute build on every restart would be worse
than asking the author to retrigger.

`down --purge` deletes the bare clones and the poller watermarks. The next start
re-clones both repos, and the poller only looks back
`POLL_INITIAL_LOOKBACK_MINUTES`, so trigger comments older than that window are
missed. Use plain `down` unless you specifically want a clean slate.

### Mirrors

Everything defaults to Tencent Cloud mirrors, matching the rest of the project.

Two separate layers, both configured in `.env`:

| Scope | Variable | Default |
|-------|----------|---------|
| Builds the agent performs (core / perception / drivers) | `MIRROR` | `tencent` |
| The agent's own image — base image | `MIRROR_BASE` | `mirror.ccs.tencentyun.com` |
| The agent's own image — PyPI | `PYPI_MIRROR` | `https://mirrors.tencentyun.com/pypi/simple/` |
| The agent's own image — apt | `APT_MIRROR` | `mirrors.tencentyun.com` |
| QEMU binfmt image | `BINFMT_IMAGE` | `mirror.ccs.tencentyun.com/tonistiigi/binfmt` |

`MIRROR` is passed to the repos' build scripts both as an env var and as
`--mirror tencent`, so their interactive mirror prompt never fires — which
matters because that prompt defaults to *tuna*, not tencent, when it cannot
read a TTY.

For a host outside the Tencent VPC, uncomment the override block in `.env`.

### Setup notes

`GITHUB_TOKEN` — a classic PAT with the `repo` scope covers both repos
(read PRs, post/edit comments, add reactions). Create one at
https://github.com/settings/tokens.

`GITHUB_WEBHOOK_SECRET` — only needed if you enable the webhook. Polling
ignores it.

Repos are cloned over **HTTPS**, so no SSH key is needed on the host. Both
repos are public; the token is used for the API, not for cloning.

The container mounts the Docker socket, which is root-equivalent access to the
host. Keep `.env` root-readable only — it holds registry and API credentials.

`RESOURCE_CENTER_API_KEY` is intentionally left unset. The build scripts
auto-register successful builds into the Resource Center image catalog, and
their interactive "sync?" confirmation defaults to *yes* whenever it cannot
read a TTY — which is always, in a container. Setting it here would silently
publish every PR build, including unreviewed and unmerged code, into the
catalog that production deployments draw from.

## Monitoring

The dashboard's Overview tab is the usual way in. For scripting, or to check
liveness without a browser:

```bash
curl -s http://localhost:25000/api/status | python3 -m json.tool
```

With polling there is no inbound traffic to confirm the agent is alive, so
`poller.last_poll_at` should be within one interval and `poller.last_error`
should be null. The dashboard flags a stale poller automatically.

## Layout

```
agents/pr_review/
  server.py            FastAPI app, static mount, lifespan wiring
  config.py            Environment configuration
  models.py            Job model, error types, command parsing
  store.py             SQLite history, build-log files, prune, reconcile
  poller.py            Polling loop + watermark persistence
  router_api.py        /api endpoints
  router_webhook.py    Optional webhook receiver
  trigger.py           Shared job creation (poll and webhook)
  job_queue.py         Async queue + worker pool
  worker.py            Pipeline, timeout, retry policy
  git_workspace.py     Bare clones, worktrees, diffs
  build_detector.py    Changed files → build targets
  builder.py           Invokes the repos' build scripts, streams logs
  reviewer.py          Deterministic checks (size, infra, secrets) + LLM helpers
  review_agent.py      The review loop
  tools.py             Sandboxed list_dir/read_file/grep/file_diff
  components.py        Component -> rules, docs, reference implementations
  rules/               Review standards as editable markdown
    common.md            infrastructure tiers, file size, COS convention
    driver.md            plugin contract, driver.yaml, renderer formats
    core.md              agent-core subsystems and their failure modes
    perception.md        the ASR audio contract
  comments.py          PR comment formatting
  web/
    index.html
    css/style.css      Redeclares agent-core's design tokens
    js/api.js          fetch helpers, escaping, formatting
    js/views.js        overview / history / detail renderers
    js/app.js          tab routing, polling loops, log tailing

deploy/pr-review/
  docker-compose.yml
  deploy.sh
  .env.example

PR_REVIEW_AGENT.md     this document (repo root, so it is findable)
```

`agents/` is a namespace for operational agents; `pr_review` is the first.
Additional agents can live alongside it as independent apps.
