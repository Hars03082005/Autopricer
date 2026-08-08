# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# PriceRef ML API — FastAPI + CatBoost/LightGBM/XGBoost ensemble
#
# Build from the REPOSITORY ROOT, not from backend/:
#   docker build -f backend/Dockerfile -t priceref-backend .
#
# Ships exactly one model variant (variant_1, ~66 MB). See .dockerignore.
# ─────────────────────────────────────────────────────────────────────────────

# ══ Stage 1: build the virtualenv ════════════════════════════════════════════
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1

# libgomp1 is needed HERE as well as in the runtime stage, not instead of it.
# The import check at the end of this stage actually loads LightGBM and XGBoost,
# and loading them dlopens libgomp.so.1 — so without this the check fails with
# the very error it exists to catch, and the build never reaches runtime.
# Nothing from this stage but /opt/venv is copied forward, so this costs the
# final image nothing.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# requirements.lock is wheel-hash pinned, so nothing is compiled from source and
# this stage needs no C toolchain.
COPY backend/requirements.lock /tmp/requirements.lock

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade "pip>=24.0"
RUN /opt/venv/bin/pip install --require-hashes --only-binary=:all: -r /tmp/requirements.lock

# CatBoost hard-depends on matplotlib/plotly/graphviz for its plotting helpers,
# which cost ~250 MB and are unreachable from an inference-only code path — but
# only if CatBoost imports them lazily, which varies by release. Left OFF by
# default so a first build cannot fail on it. Turn on with:
#   docker build --build-arg SLIM_ML=1 ...
# The import check below is the gate: if slimming breaks a needed import, the
# build fails here rather than at model load in production.
ARG SLIM_ML=0
RUN if [ "$SLIM_ML" = "1" ]; then \
      /opt/venv/bin/pip uninstall -y \
        matplotlib plotly graphviz pillow fonttools contourpy kiwisolver \
        cycler pyparsing narwhals; \
      echo "SLIM_ML=1: removed CatBoost plotting dependencies"; \
    else \
      echo "SLIM_ML=0: keeping full dependency set"; \
    fi

# Drop bundled test suites (never imported, tens of MB across numpy/scipy/pandas).
RUN find /opt/venv -type d -name tests -prune -exec rm -rf {} + || true

# Verify every module the serving path imports is present and loadable.
RUN /opt/venv/bin/python -c "import catboost, lightgbm, xgboost, pandas, numpy, sklearn, scipy, joblib, fastapi, uvicorn, httpx, jwt; print('import check ok | catboost', catboost.__version__, '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__)"


# ══ Stage 2: runtime ═════════════════════════════════════════════════════════
FROM python:3.11-slim-bookworm AS runtime

# libgomp1 = OpenMP runtime. LightGBM, XGBoost and CatBoost all link against it
# and python:*-slim omits it — without this the image builds fine and then dies
# at model load with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Pin the serving variant. Must match the variant directory copied below;
# docker-entrypoint.sh aborts startup if the two disagree.
ENV ACTIVE_VARIANT_ID=variant_1

# Single-threaded BLAS/OpenMP. backend/main.py sets these at import time too,
# but they must be in the environment before numpy is first imported to actually
# take effect. A single prediction is milliseconds; concurrency comes from
# replicas, and unbounded thread pools inside a CPU-capped container cause more
# contention than they relieve.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CUDA_VISIBLE_DEVICES=""

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Model artifacts: variant_1 is the active default (MAPE 7.48%); variant_2 and
# variant_3 ship alongside it so the Results-screen variant switcher works on
# the deployed server without a 404. Total: ~198 MB across all three.
# Explicit paths so a stale .dockerignore can never silently pull in
# unintended or archived variants.
COPY model_registry/registry.json  /app/model_registry/registry.json
COPY model_registry/variant_1/     /app/model_registry/variant_1/
COPY model_registry/variant_2/     /app/model_registry/variant_2/
COPY model_registry/variant_3/     /app/model_registry/variant_3/

# Application code — smallest and most frequently changed, so it goes last.
COPY backend/            /app/backend/
COPY scripts/            /app/scripts/

# Non-root. Nothing under /app needs to be writable: the one code path that
# writes (model_registry.activate_variant -> registry.json) is refused in
# containers via ALLOW_RUNTIME_VARIANT_SWITCH, because a disk write inside one
# replica silently desyncs it from every other replica behind the same ingress.
# chmod +x is explicit rather than inherited from the host file mode: this repo
# is developed on Windows, where git does not reliably carry the execute bit.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && chown -R root:root /app \
 && chmod -R a-w,a+rX /app \
 && chmod 0555 /app/backend/docker-entrypoint.sh
USER 10001

EXPOSE 8000

# Generous start-period: loading the ensemble off disk into RAM takes tens of
# seconds on a cold container. curl/wget are absent from slim images, so the
# probe is a Python script.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD ["python", "/app/backend/healthcheck.py"]

ENTRYPOINT ["/app/backend/docker-entrypoint.sh"]
