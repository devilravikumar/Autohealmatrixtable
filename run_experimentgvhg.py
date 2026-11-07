#!/usr/bin/env python3
"""
AutoHeal-RAG evaluation harness (SIMULATION).

What this is: a seeded fault-injection simulator of a 12-service REST system
(service names mirror Train-Ticket), producing labeled REST API error records,
plus non-LLM implementations of four diagnosis systems:
  rule_based, rag, agentic_rag, autoheal_rag
All metrics are computed from the raw per-error logs this script writes.

What this is NOT: it is not the real Train-Ticket deployment, and it uses no LLM.
"Retrieval" is TF-IDF over a separate troubleshooting knowledge base; "self-reflection"
is a re-scoring step using runtime metrics. Numbers reflect this simulator only.
Timing is measured wall-clock of this code, not LLM latency.
"""
import json, os, re, sys, time, random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/autoheal_exp/out"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42]
os.makedirs(OUT, exist_ok=True)

SERVICES = ["ts-auth-service", "ts-user-service", "ts-order-service", "ts-travel-service",
            "ts-route-service", "ts-station-service", "ts-ticket-office-service",
            "ts-payment-service", "ts-inside-payment-service", "ts-seat-service",
            "ts-price-service", "ts-notification-service"]
CLASSES = ["db_connection_timeout", "auth_token_expired", "downstream_unavailable",
           "rate_limit_exceeded", "memory_exhaustion", "config_missing",
           "schema_mismatch", "bad_request_validation"]
# remediation action per class, and risk level (risky actions need high confidence)
ACTION = {"db_connection_timeout": ("reset_connection_pool", "risky"),
          "auth_token_expired": ("refresh_token", "low"),
          "downstream_unavailable": ("restart_downstream", "risky"),
          "rate_limit_exceeded": ("backoff_and_retry", "low"),
          "memory_exhaustion": ("scale_out_instance", "risky"),
          "config_missing": ("reload_config", "low"),
          "schema_mismatch": ("rollback_deployment", "risky"),
          "bad_request_validation": ("no_op_client_fix", "low")}

# ---- test-time log templates (what the injected fault emits) ----
TEST_T = {
 "db_connection_timeout": ["HikariPool-1 - Connection is not available, request timed out after {n}ms",
   "SQLTransientConnectionException: could not acquire connection from pool within {n}ms",
   "java.net.SocketTimeoutException: Read timed out while querying {svc} database",
   "Unable to obtain JDBC connection; pool exhausted, waiting {n}ms"],
 "auth_token_expired": ["401 Unauthorized: JWT expired at {n}, token validation failed",
   "TokenExpiredException: access token is no longer valid for user {n}",
   "Authentication failed: bearer token expired, please re-authenticate",
   "io.jsonwebtoken.ExpiredJwtException: JWT signature window passed {n}s ago"],
 "downstream_unavailable": ["503 Service Unavailable: upstream {svc2} refused connection",
   "Connection refused: ts downstream {svc2} not reachable on port {n}",
   "FeignException: no instances available for {svc2}",
   "java.net.ConnectException: failed to call {svc2}, host unreachable"],
 "rate_limit_exceeded": ["429 Too Many Requests: rate limit of {n} req/s exceeded",
   "RateLimiterException: quota exceeded for client {n}, retry later",
   "Throttled: request rejected by gateway, limit {n}/min reached",
   "Too many requests from client {n}; Retry-After header set"],
 "memory_exhaustion": ["java.lang.OutOfMemoryError: Java heap space in {svc}",
   "Container {svc} killed: OOMKilled, memory limit {n}Mi reached",
   "GC overhead limit exceeded; heap usage at {n} percent",
   "Cannot allocate memory: worker process died in {svc}"],
 "config_missing": ["IllegalStateException: required property '{k}' is not set",
   "Could not resolve placeholder '{k}' in value ${{{k}}}",
   "ConfigurationException: missing environment variable {k} for {svc}",
   "BeanCreationException: configuration key {k} not found"],
 "schema_mismatch": ["JsonMappingException: unrecognized field '{k}' not marked as ignorable",
   "HttpMessageNotReadableException: cannot deserialize response, unexpected type for '{k}'",
   "Schema mismatch: expected field '{k}' missing in payload from {svc2}",
   "UnmarshalException: incompatible DTO version, field '{k}' changed"],
 "bad_request_validation": ["400 Bad Request: field '{k}' must not be null",
   "MethodArgumentNotValidException: validation failed for '{k}'",
   "ConstraintViolationException: '{k}' size must be between 1 and {n}",
   "Invalid request parameter '{k}': value rejected by validator"],
}
# generic, low-information messages shared across classes
GENERIC = ["500 Internal Server Error", "Request failed with status 500",
           "Unexpected error while processing request", "Service error: operation could not be completed",
           "Exception in thread http-nio-8080-exec: request failed"]
KEYS = ["tripId", "orderId", "seatClass", "stationName", "price", "accountId", "routeId", "token"]
# misleading phrases appended as noise (retry notes etc.)
NOISE = ["retrying request after timeout", "token refresh scheduled", "upstream health check pending",
         "limit check skipped", "memory usage reported by monitor", "config reload requested",
         "validation hook invoked", "schema registry consulted"]

# ---- knowledge base (separate paraphrased troubleshooting records) ----
KB_T = {
 "db_connection_timeout": ["database connection pool timeout acquiring connection {svc}",
   "slow query exhausted jdbc pool, connections timed out", "mysql connection wait timeout, pool starvation",
   "timeout reading from database socket, connection not available"],
 "auth_token_expired": ["expired jwt token rejected 401 unauthorized", "access token validity window ended, reauthenticate",
   "bearer token expiry causes authentication failure", "token no longer valid, refresh required"],
 "downstream_unavailable": ["dependent service unreachable connection refused 503", "no healthy instances of downstream service",
   "upstream service down, call failed host unreachable", "service discovery returned no instance for dependency"],
 "rate_limit_exceeded": ["429 too many requests throttled by gateway", "client quota exceeded, retry after delay",
   "request rate limit reached, requests rejected", "burst traffic hit rate limiter"],
 "memory_exhaustion": ["java heap space out of memory error", "container oom killed memory limit reached",
   "garbage collection overhead, heap nearly full", "process died cannot allocate memory"],
 "config_missing": ["required configuration property not set", "unresolved placeholder missing environment variable",
   "bean creation failed missing config key", "application property absent at startup"],
 "schema_mismatch": ["json deserialization failed unknown or missing field", "dto version incompatible after deployment",
   "payload does not match expected schema from service", "unexpected type when mapping response body"],
 "bad_request_validation": ["400 bad request field must not be null", "request validation failed for parameter",
   "constraint violation on input field", "client sent invalid parameter rejected by validator"],
}
KB_GENERIC = ["internal server error 500 request failed", "unexpected error processing request"]

# runtime metric signature per class: (cpu%, mem%, latency_ms, upstream_err_rate) means, shared std -> overlap
SIG = {"db_connection_timeout": (45, 55, 4200, 0.05), "auth_token_expired": (25, 45, 300, 0.02),
       "downstream_unavailable": (30, 50, 2500, 0.80), "rate_limit_exceeded": (80, 55, 400, 0.05),
       "memory_exhaustion": (70, 97, 1800, 0.10), "config_missing": (10, 30, 100, 0.01),
       "schema_mismatch": (30, 45, 250, 0.40), "bad_request_validation": (20, 40, 120, 0.01)}
SD = (14, 10, 900, 0.12)


def gen_error(rng, i):
    c = rng.choice(CLASSES)
    svc = rng.choice(SERVICES); svc2 = rng.choice([s for s in SERVICES if s != svc])
    r = rng.random()
    if r < 0.30:                       # generic uninformative message
        msg = rng.choice(GENERIC)
        kind = "generic"
    else:
        msg = rng.choice(TEST_T[c]).format(n=rng.randint(10, 9999), svc=svc, svc2=svc2, k=rng.choice(KEYS))
        kind = "specific"
        if rng.random() < 0.35:        # misleading noise from a random (often wrong) class vocabulary
            msg += " | " + rng.choice(NOISE)
    m = [max(0, rng.gauss(SIG[c][j], SD[j])) for j in range(4)]
    m[1] = min(m[1], 100); m[3] = min(m[3], 1.0)
    transient = rng.random() < 0.15    # fault would self-heal without intervention
    return {"id": i, "service": svc, "true_class": c, "message": msg, "msg_kind": kind,
            "cpu": round(m[0], 1), "mem": round(m[1], 1), "latency_ms": round(m[2]),
            "upstream_err": round(m[3], 3), "transient": transient}


def build_kb(rng):
    docs, labels, metrics = [], [], []
    for c in CLASSES:
        for _ in range(80):
            if rng.random() < 0.10:
                docs.append(rng.choice(KB_GENERIC))
            else:
                docs.append(rng.choice(KB_T[c]).format(svc=rng.choice(SERVICES)) + " " + rng.choice(["", "", rng.choice(NOISE)]))
            labels.append(c)
            metrics.append([max(0, rng.gauss(SIG[c][j], SD[j])) for j in range(4)])
    return docs, np.array(labels), np.array(metrics)


def clean_query(t):
    t = re.sub(r"\b\d+\b", " ", t); t = re.sub(r"ts-[a-z-]+", " ", t)
    return t.split("|")[0]


class Systems:
    def __init__(self, kb_docs, kb_labels, kb_metrics):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        self.K = self.vec.fit_transform(kb_docs)
        self.labels = kb_labels
        self.mu = {c: kb_metrics[kb_labels == c].mean(0) for c in CLASSES}
        self.sd = np.array(SD)  # estimated metric spread assumed known to the operator
        self.kb_metrics = kb_metrics

    # --- 1. rule-based: ordered regex rules, first match wins ---
    RULES = [("auth_token_expired", r"jwt|token"), ("rate_limit_exceeded", r"429|rate limit|too many|quota|throttl"),
             ("memory_exhaustion", r"outofmemory|oom|heap|memory"), ("downstream_unavailable", r"503|refused|unreachable|no instances"),
             ("db_connection_timeout", r"timed? out|timeout|jdbc|pool|connection"), ("config_missing", r"property|placeholder|environment variable|configuration"),
             ("schema_mismatch", r"deserializ|unrecognized field|schema|dto|unmarshal"), ("bad_request_validation", r"400|valid|constraint|invalid")]

    def rule_based(self, e):
        t = e["message"].lower()
        for c, pat in self.RULES:
            if re.search(pat, t): return c, 1.0
        return "db_connection_timeout", 0.0   # default fallback

    def _scores(self, q, k=5):
        sims = (self.vec.transform([q]) @ self.K.T).toarray()[0]
        idx = np.argsort(-sims)[:k]
        s = {c: 0.0 for c in CLASSES}
        for j in idx: s[self.labels[j]] += sims[j]
        return s, sims[idx[0]]

    @staticmethod
    def _norm(s):
        tot = sum(s.values())
        return {c: (v / tot if tot > 0 else 1 / len(CLASSES)) for c, v in s.items()}

    # --- 2. conventional RAG: top-5 retrieval, argmax, no reflection/gating ---
    def rag(self, e):
        s, _ = self._scores(e["message"]); p = self._norm(s)
        c = max(p, key=p.get); return c, p[c]

    # --- 3. agentic RAG: reformulate query when retrieval is weak, merge evidence ---
    def agentic(self, e):
        s, top = self._scores(e["message"]); p = self._norm(s)
        if top < 0.35 or max(p.values()) < 0.6:
            s2, _ = self._scores(clean_query(e["message"]) + " " + e["service"].replace("ts-", "").replace("-", " "))
            p2 = self._norm(s2)
            p = self._norm({c: p[c] + p2[c] for c in CLASSES})
        c = max(p, key=p.get); return c, p[c], p

    # --- 4. AutoHeal-RAG: agentic + confidence + self-reflection (<0.85) + policy gate ---
    def metric_lik(self, e):
        x = np.array([e["cpu"], e["mem"], e["latency_ms"], e["upstream_err"]])
        ll = {c: -0.5 * np.sum(((x - self.mu[c]) / self.sd) ** 2) for c in CLASSES}
        m = max(ll.values()); w = {c: np.exp(v - m) for c, v in ll.items()}
        return self._norm(w)

    def autoheal(self, e):
        c, conf, p = self.agentic(e)
        reflected = False
        if conf < 0.85:                       # self-reflection: fuse text posterior with runtime-metric likelihood
            reflected = True
            ml = self.metric_lik(e)
            p = self._norm({k: p[k] * ml[k] for k in CLASSES})
            c = max(p, key=p.get); conf = p[c]
        act, risk = ACTION[c]
        thr = 0.85 if risk == "risky" else 0.60
        allowed = conf >= thr                 # policy gate
        return c, conf, reflected, allowed


def simulate_remediation(rng, e, diagnosed, act_applied):
    """Return (healthy_at_60s, seconds_to_health, unnecessary)."""
    t_self = rng.uniform(5, 40) if e["transient"] else None
    if not act_applied:
        healthy = t_self is not None and t_self <= 60
        return healthy, t_self, False
    correct = ACTION[diagnosed][0] == ACTION[e["true_class"]][0]
    t_fix = rng.uniform(3, 20) if correct else None
    cands = [t for t in (t_fix, t_self) if t is not None]
    t = min(cands) if cands else None
    healthy = t is not None and t <= 60
    unnecessary = e["transient"] or (not correct)
    return healthy, t, unnecessary


def run_seed(seed):
    rng = random.Random(seed)
    kb = build_kb(random.Random(seed + 1000))
    S = Systems(*kb)
    errors = [gen_error(rng, i) for i in range(N)]
    rows = {k: [] for k in ["rule_based", "rag", "agentic_rag", "autoheal_rag"]}
    for e in errors:
        for name in rows:
            t0 = time.perf_counter()
            reflected = False; allowed = True
            if name == "rule_based": c, conf = S.rule_based(e)
            elif name == "rag": c, conf = S.rag(e)
            elif name == "agentic_rag": c, conf, _ = S.agentic(e)
            else: c, conf, reflected, allowed = S.autoheal(e)
            dt = (time.perf_counter() - t0) * 1000
            rr = random.Random(seed * 100003 + e["id"])   # same fault dynamics for every system
            healthy, t_h, unnec = simulate_remediation(rr, e, c, allowed)
            rows[name].append({"id": e["id"], "true": e["true_class"], "pred": c, "conf": round(float(conf), 4),
                               "reflected": reflected, "action_applied": allowed, "healthy_60s": healthy,
                               "unnecessary_action": unnec, "transient": e["transient"], "diag_ms": round(dt, 3)})
    return errors, {k: pd.DataFrame(v) for k, v in rows.items()}


def metrics(df):
    y, p = df["true"], df["pred"]
    acc = (y == p).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=CLASSES, average="macro", zero_division=0)
    applied = df[df.action_applied]
    return {"root_cause_accuracy": acc, "diagnostic_precision_macro": prec, "recall_macro": rec, "f1_macro": f1,
            "remediation_success_over_all_errors": df["healthy_60s"].mean(),
            "remediation_success_over_applied_actions": applied["healthy_60s"].mean() if len(applied) else float("nan"),
            "actions_applied": int(df.action_applied.sum()), "actions_withheld_by_gate": int((~df.action_applied).sum()),
            "unnecessary_remediations": int((df.action_applied & df.unnecessary_action).sum()),
            "false_diagnoses": int((y != p).sum()),
            "reflection_rate": df["reflected"].mean(), "mean_diag_ms": df["diag_ms"].mean()}


def main():
    all_m = {}
    for seed in SEEDS:
        t0 = time.time()
        errors, dfs = run_seed(seed)
        sd = os.path.join(OUT, f"seed_{seed}"); os.makedirs(sd, exist_ok=True)
        with open(os.path.join(sd, "injected_errors_ground_truth.jsonl"), "w") as f:
            for e in errors: f.write(json.dumps(e) + "\n")
        m = {}
        for name, df in dfs.items():
            df.to_csv(os.path.join(sd, f"predictions_{name}.csv"), index=False)
            m[name] = metrics(df)
            cm = confusion_matrix(df["true"], df["pred"], labels=CLASSES)
            pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(
                os.path.join(sd, f"confusion_matrix_{name}.csv"))
        r, a = m["rag"], m["autoheal_rag"]
        m["derived"] = {
            "accuracy_gain_over_rag_pts": 100 * (a["root_cause_accuracy"] - r["root_cause_accuracy"]),
            "unnecessary_remediation_reduction_vs_rag_pct": 100 * (1 - a["unnecessary_remediations"] / max(1, r["unnecessary_remediations"])),
            "false_diagnosis_reduction_vs_rag_pct": 100 * (1 - a["false_diagnoses"] / max(1, r["false_diagnoses"])),
        }
        json.dump(m, open(os.path.join(sd, "metrics.json"), "w"), indent=2, default=float)
        all_m[seed] = m
        print(f"seed {seed} done in {time.time()-t0:.1f}s")
    json.dump(all_m, open(os.path.join(OUT, "metrics_all_seeds.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
