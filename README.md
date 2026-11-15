# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/hiduplahzomb/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/hiduplahzomb/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator: one server speaking the OpenAI and Anthropic APIs, routing each request to the best free model that can serve it across ten simulated free-tier providers — ranked by quality, planned under each provider's budget, with fallback chains when a provider refuses.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL` (the env serving the providers) and `MAKEFUN_PROMPT_FILE` (the brief: the provider directory, the quality table and the workload manifest). `node main.ts --list` prints the catalog and exits.

## How it routes

- **Ranked by quality.** Every free model meeting a request's requirements (tools, vision, JSON mode, context window) is ranked by the quality table's `rating_milli`; the best serves.
- **Planned under budget.** The whole 20-request workload is planned up front, each provider capped at the minimum of its published per-trial budget range, so the plan never assumes more than a provider is guaranteed to serve. Weight-2 requests come first in the manifest, so they get the best models.
- **Failover with a circuit breaker.** A refusal (402, 429, 5xx) is answered by falling back down the rating-ordered chain, preferring providers still under their planned cap. A provider whose budget is spent or that is in an outage is retired at once (its refusal names the cause); a schedule refusal counts toward a circuit breaker that takes a provider out after two consecutive failures. There is no same-provider retry: a fresh provider's first call always succeeds, so falling back spends fewer calls than retrying.
- **Every wire, both ways.** OpenAI and Anthropic clients, each to openai, anthropic and gemini providers, JSON and streaming — with the providers' echo (system, user, tools, max_tokens, stops, image, JSON mode) preserved exactly and the providers' own usage and stop family reported back.
- **One successful call per request.** The attempt loop stops at the first success; the tie-break is the number of upstream calls, so no call is spent twice.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, work poll.
- `lib/providers.ts` — the brief's parser, the auth headers, the catalog probe.
- `lib/router.ts` — the rating-ranked candidates, the up-front plan, the failover policy.
- `lib/wire.ts` — the client and upstream wires: request and response translation, SSE.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ykaz8n5y&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
