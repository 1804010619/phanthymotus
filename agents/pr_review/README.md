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
| `{provider}/{model}/**` | `build.sh {provider}/{model}` |
| `dji/base/**` | none — base image, built manually |
| `build.sh`, `README*` | none — review only |

The agent invokes the repos' existing build scripts rather than reimplementing
the build. Image tags, mirrors, and registry handling stay in one place.

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

Two phases.

**Rule checks** — deterministic, no LLM:

- Dockerfile modified (minimal-change principle)
- Files over 1MB added
- Possible secrets (`.env`, `credentials`, `secret`, `.pem`, `.key`;
  `*.example` and `*.sample` are exempt)

**LLM review** — one call to an OpenAI-compatible `/v1/chat/completions`
endpoint, with the project's architecture and review rules in the system
prompt, and the rule-check findings passed in as context. Output is English,
structured as Summary / Issues / Suggestions.

This is a single call, not an agent loop: the pipeline is deterministic, so
there is nothing for a loop to decide. Diffs are truncated to
`MAX_DIFF_LINES`. If the LLM is unconfigured or fails, the build result is
still posted and the review section says so.

## Deploy

```bash
cd phanthymotus/deploy/pr-review
cp .env.example .env
$EDITOR .env          # GITHUB_TOKEN, REGISTRY_*, LLM_*
./deploy.sh
```

`deploy.sh` validates the required `.env` values, registers QEMU for ARM64
cross-compilation, then builds and starts the container.

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

## Monitoring

With polling there is no inbound traffic to confirm liveness, so check the
status endpoint:

```bash
curl -s http://localhost:15690/status | python3 -m json.tool
```

`poller.last_poll_at` should be within one interval, and `poller.last_error`
should be null. Also available:

| Endpoint | Purpose |
|----------|---------|
| `GET /status` | Queue depth, active jobs, poller health, effective config |
| `GET /jobs` | Recent jobs with status and attempt count |
| `GET /jobs/{id}` | One job: build results, image tags, logs, review text |

## Layout

```
agents/pr_review/
  server.py            FastAPI app + lifespan wiring
  config.py            Environment configuration
  models.py            Job model, error types, command parsing
  poller.py            Polling loop + watermark persistence
  router_webhook.py    Optional webhook receiver
  router_status.py     Status / jobs endpoints
  trigger.py           Shared job creation (poll and webhook)
  job_queue.py         Async queue + worker pool
  worker.py            Pipeline, timeout, retry policy
  git_workspace.py     Bare clones, worktrees, diffs
  build_detector.py    Changed files → build targets
  builder.py           Invokes the repos' build scripts
  reviewer.py          Rule checks + LLM review
  comments.py          PR comment formatting

deploy/pr-review/
  docker-compose.yml
  deploy.sh
  .env.example
```

`agents/` is a namespace for operational agents; `pr_review` is the first.
Additional agents can live alongside it as independent apps.
