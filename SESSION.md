# Session record — CI/CD and Azure deployment

**Date:** 2026-08-03
**Outcome:** PriceRef containerised and deployed to Azure Container Apps, CI and CD both green.

This file is a working record so the context survives a folder move or a new
session. It is not part of the application; delete it whenever it stops being
useful.

---

## 1. Live deployment

**Public URL**
<https://priceref-staging-frontend.calmbay-50792673.centralindia.azurecontainerapps.io>

Verified end to end (all 12 smoke checks pass, run from Windows against the live
site): SPA served, runtime config injected at boot, backend reachable through the
nginx proxy, `variant_1` loaded, real valuation returned (₹13,37,500 for the test
2021 Honda City), anonymous callers refused 401, variant activation refused 403,
security headers present.

### Environment naming — read this first

There is **one** deployed environment and it is called **`staging`**. That was a
deliberate choice, not an oversight: a single environment was wanted, and using
the `staging` name meant no workflow changes, since push-to-`main` already
auto-deploys staging while the production job stays behind a manual gate.

Consequence: **the live application is named `staging` everywhere** —
`priceref-staging-frontend`, `priceref-staging-backend`, `pricerefacrstaging`.
The Supabase project backing it is named `PriceRefPES`.

`production` is unconfigured by design: the GitHub environment was deleted, it
has no secrets, and `deploy-production` only runs on an explicit
`workflow_dispatch` with `promote_to_production` ticked. Nothing can reach it
accidentally.

---

## 2. Repositories and remotes

| Remote | URL | Note |
| :--- | :--- | :--- |
| `priceref` | <https://github.com/srinvaid/PriceRefPES> | **The working repo.** Private. CI/CD runs here. |
| `origin` | <https://github.com/UmaDamotharan/Price-Prediction> | Upstream. Push access only, no admin — cannot configure CI/CD there. |

Branches: `main` (deploys), `feature` (work branch). The new repo was created
because configuring environments, secrets and OIDC needs admin, which the
upstream does not grant.

---

## 3. Azure

| Item | Value |
| :--- | :--- |
| Subscription | `Azure subscription 1` — `951eee0a-1e59-4b29-85f2-584cafe33747` |
| Tenant | `a9ae2ccb-d547-4920-84c0-1396a178531d` (ForgePointIndia) |
| Signed-in user | `srinvaid@ForgePointIndia.onmicrosoft.com` — **Owner** on the subscription |
| Resource group | `rg-priceref` |
| Region | `centralindia` |
| Registry | `pricerefacrstaging` (Basic, admin user disabled) |
| Entra app | `priceref-github` — appId `99a13019-e2f1-4754-a1d5-54529e35735e` |
| Roles on RG | **Contributor** *and* **Role Based Access Control Administrator** |

No client secret exists anywhere — authentication is OIDC federation, so there is
nothing to rotate or leak. The IDs above are identifiers, not credentials.

### The OIDC subject format — the single least obvious thing here

GitHub does **not** present the documented `repo:<owner>/<repo>:...` subject. It
embeds immutable numeric IDs:

```
repo:srinvaid@18724072/PriceRefPES@1321439955:environment:staging
```

owner id `18724072`, repo id `1321439955`. Get them with:

```bash
gh api repos/<owner>/<repo> --jq '{repo: .id, owner: .owner.id}'
```

Six federated credentials are registered: three using this immutable form (the
ones that actually work) and three legacy name-based ones kept as fallback. A
name-based-only setup fails every login with `AADSTS700213`, whose error text
helpfully quotes the exact subject GitHub sent.

Because the IDs are immutable, they survive repository renames — which matters,
since this repo was renamed from `priceref` to `PriceRefPES` mid-setup.

---

## 4. GitHub configuration

**Repository variables** (Settings → Actions → Variables):
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`

**Environment secrets** on `staging` (six):
`AZURE_RESOURCE_GROUP`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`,
`SUPABASE_PROJECT_REF`, `SUPABASE_DB_PASSWORD`, `SUPABASE_ACCESS_TOKEN`

`SUPABASE_JWT_SECRET` is deliberately **unset** — the project uses asymmetric JWT
keys, which `backend/auth.py` verifies against the published JWKS. It is only
needed by projects still on legacy HS256 signing.

**Production gating.** Required-reviewer protection is unavailable on private
repositories under GitHub Free (the API returns 422). Production is therefore
gated on manual `workflow_dispatch` instead. If the account is ever upgraded, add
the reviewer rule and drop the `if:` on `deploy-production`.

---

## 5. Local tooling

**Azure CLI 2.88.0 lives at `C:\Users\srinv\.azcli-venv`** — installed via
`pip install azure-cli` into a Python 3.11 venv, because the winget MSI is
machine-scope and fails with error 1603 in a non-elevated shell.

```powershell
& "C:\Users\srinv\.azcli-venv\Scripts\az.bat" <command>
```

This is **outside the project folder**, so moving the project does not affect it.
Same for `az login` state (`~/.azure`) and `gh` auth.

---

## 6. Defects found and fixed

All sixteen were first-execution failures: the pipeline was structurally sound
but had never run against a real subscription, so every boundary with an external
system was unverified.

### CI

1. **`ruff` S310** in the new health probe — lint red from the first run.
2. **Lockfile resolved against the build host, not the target.** `pip --platform`
   selects wheels but still evaluates environment markers against the machine
   running pip. Generated on Windows the lock gained `colorama` and lost `uvloop`
   *and* `nvidia-nccl-cu12`. Since the Dockerfile installs with
   `--require-hashes`, the missing transitive dep meant **the image could never
   have built**. Now resolved with `uv --python-platform`, pinned.
3. **`*.lock` not pinned to LF** in `.gitattributes` — CRLF breaks the
   backslash line-continuations pip relies on.
4. **`libgomp1` only in the runtime stage** — the builder's import check died on
   exactly the error that install exists to prevent.
5. **`docker images` given two repository arguments** — accepts at most one; an
   informational size report was failing the whole job.
6. **`aquasecurity/trivy-action@0.28.0`** — that tag has never existed (the
   action publishes `v`-prefixed tags). Job died at action resolution.

### Setup documentation

7. **Contributor is not sufficient.** `infra/main.bicep` creates the `AcrPull`
   role assignment, which needs `Microsoft.Authorization/roleAssignments/write`.
   The first deploy would have failed `AuthorizationFailed`.
8. **The production gate did not exist.** It relied on required reviewers, which
   GitHub silently refuses to create on this plan — so every push to `main` would
   have deployed straight to production.

### Deployment

9. **OIDC subject format** — see section 3.
10. **Empty optional secret.** Container Apps rejects a declared secret with an
    empty value and fails the *entire* deployment. `supabaseJwtSecret` is
    legitimately blank here, so the template turned "not needed" into a hard
    failure. Optional secrets and their env refs are now built conditionally.
11. **Apps provisioned before their images existed.** The build job ran the full
    Bicep template so the registry would exist — but the template also declares
    both container apps, referencing SHA tags not pushed until several steps
    later. `MANIFEST_UNKNOWN`. The registry is now created alone via idempotent
    `az acr create`.
12. **nginx sent no SNI upstream** — `502` on *every* proxied API call. ACA's
    internal ingress routes by SNI and nginx does not send it by default for an
    https `proxy_pass`. **This class of bug is structurally invisible to
    `docker compose`, which proxies over plain HTTP.**

### Security and tooling

13. **All three security headers were silently absent.** nginx's `add_header`
    does not merge across levels: a location declaring any `add_header` discards
    every inherited one. Every location sets its own `Cache-Control`, so the
    server-level security headers never applied anywhere. Verified against the
    live site before and after.
14. **The smoke test checked one of the three** and then reported "security
    headers present" — understating a failure that had removed all three.
15. **`smoke-test.sh` hardcoded `python3`** — on Windows both `python` and
    `python3` resolve to the Microsoft Store alias stub, which satisfies
    `command -v` and then fails when run. The script is meant to be runnable by
    hand against a misbehaving deployment; it could only run in CI. Now probes
    `python3` → `python` → `py -3` by executing each.
16. **Sign-up confirmation emails pointed at `http://localhost:3000`.**
    `signUp()` was called without `emailRedirectTo`, so Supabase fell back to the
    project Site URL, still on its factory default. Now passes
    `window.location.origin`.

---

## 7. Outstanding — nothing blocking

### Needs a decision

- **`/health` reports two meaningless fields.** `ensemble_enabled` reads
  `meta["ensemble"]["enabled"]`, a key this metadata format does not have, so it
  is *always* `false`. `model_name` falls back to a hardcoded
  `"CatBoostRegressor"` for the same reason.
  **The ensemble is genuinely running** — `ensemble_predictor.py:32` falls back
  correctly to `metadata["ensemble_weights"]`, and live `/predict` responses
  include `ensemble_variance`, which is impossible from a single model. The
  README is accurate; only the health reporting is wrong. Worth fixing so it
  reports the predictor's resolved state.
- **Trivy SARIF upload is `continue-on-error`** — added defensively on the
  assumption it would 403 on a private Free repo. It actually succeeds, so the
  guard now only hides genuine failures. Safe to remove.

### Worth doing

- **`xgboost-cpu`** would drop ~300 MB from the 1.81 GB backend image.
  `nvidia-nccl-cu12` is a GPU collective-communications library Container Apps
  has no GPU to use, pulled in by `xgboost` — which carries **0.00%** ensemble
  weight. Needs the model tests run against it.
- **`ml_training/requirements.txt` is completely unpinned.** Combined with the
  artifacts carrying no record of their training versions, a retrain may not
  reproduce the current weights even with seeds fixed at 42.
- **Supabase email is rate-limited** (~2/hour on the built-in sender, testing
  only). Configure custom SMTP before real users. Email confirmation may
  currently be disabled for testing — check before going live.

---

## 8. Model rebuild notes

Everything needed to rebuild is in the repo; **nothing is proprietary to the
original authors except the raw dataset**.

- Training source: `ml_training/clean-1.py` → `train-1.py`
- Data: `ml_training/data/overall.csv` (raw) → `processed_overall.csv` (33,979 rows)
- Dependencies: pandas, numpy, catboost, lightgbm, xgboost, scipy, scikit-learn
  — all permissive open source (Apache 2.0 / MIT / BSD-3)
- Artifacts (`.cbm`, `.txt`, `.json`, `.pkl`) are **all generated** by
  `train-1.py`. The pickles were verified to contain only plain primitives — no
  custom classes, so they are trivially convertible to JSON.
- Ensemble weights live in `model_registry/variant_1/model_metadata.json`:
  LightGBM **89.18%**, CatBoost **10.82%**, XGBoost **0.00%**.

**`registry.json` is state, not a derived artifact.** It is read by the trainer
(to pick the next variant id and the best-MAPE default), written by
`register_variant()`, and read by the backend at runtime — it is `COPY`'d into
the image. It also holds the S5 routing *behaviour* under `variant_4`
(`s5_max_age`, `activation_condition`), so deleting it silently disables the S5
specialist rather than erroring.

**Retraining does not overwrite `variant_1`.** `next_variant_id()` returns the
first id where *neither* the directory *nor* a `registry.json` entry exists — so
with 1–4 present you get `variant_5`. To genuinely regenerate `variant_1` you
must delete both the directory and its JSON block. Note that `register_variant()`
auto-promotes whichever variant has the best MAPE, rewriting `"default"`.

---

## 9. After moving the folder to `C:\github`

The move is low-risk. Nothing in the repo uses absolute paths, and git remotes
are stored as URLs in `.git/config`.

Unaffected (all live outside the project folder): the Azure CLI venv, `az login`
state, `gh` auth, and the global git config.

Worth doing once after the move:

```bash
cd C:/github/POCars-PES          # or whatever you name it
git remote -v                    # expect origin + priceref
git status                       # expect clean
./scripts/smoke-test.sh "https://priceref-staging-frontend.calmbay-50792673.centralindia.azurecontainerapps.io"
```

`.claude/settings.local.json` travels with the folder and keeps the
`Bash(gh pr merge *)` permission rule. Claude Code's per-project memory is keyed
to the directory path, so a session in the new location starts without this
session's history — which is what this file is for.
