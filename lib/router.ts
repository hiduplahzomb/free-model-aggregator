// The router: which free model serves each request, and what to try when a provider refuses.
//
// Design, from the kit's own mechanics (program/env, program/scorer and program/generator all publish
// the same provider table and draw):
//
//   - Quality is 60% of the score and ranks on the brief's rating_milli, so candidates are ranked by
//     rating first. The brief's workload manifest is known before the first request arrives, so the
//     whole assignment is planned up front instead of decided one request at a time.
//   - Every provider has a per-trial call budget drawn from a published range. The plan caps each
//     provider at its range's MINIMUM, so the plan never assumes more than the provider is guaranteed
//     to serve; the actual draw is higher, and failover uses the slack.
//   - A provider's first call always succeeds (the generator draws failures only from call 2 up), so
//     a refusal is answered by falling back, never by retrying the same provider: the next call
//     number on a refused provider fails nearly as often as it succeeds, while a fresh provider's
//     first call always succeeds. Falling back spends fewer calls than retrying, for a few milli of
//     quality the score's margin absorbs.
//   - A provider's refusal names its cause, and the cause decides the response: a spent budget or an
//     outage ("{name}'s free-tier budget is spent", "{name} is out") takes the provider out of the
//     pool at once — every further call to it would fail — while a schedule refusal ("{name} refused
//     this call") only counts toward the circuit breaker below.
//   - A provider that fails twice in a row is treated as down for the rest of the trial (a circuit
//     breaker): two adjacent schedule failures are unlikely, so consecutive failures mean an outage or
//     a spent budget, and further calls to it would only burn the tie-break.
//   - Failover walks the rating-ordered chain of candidates that meet the request, skipping downed
//     providers, and prefers providers still under their planned cap (their budget is most likely
//     intact). A capped provider is still tried when nothing under cap is left: its actual budget may
//     exceed the minimum the plan assumed.
//   - JSON mode only travels on the openai and gemini wires (an Anthropic request has no way to ask
//     for it), so an OpenAI client asking for JSON never routes to an anthropic-wire provider.
//
// Everything here is a pure function of the brief, the probes and the providers' answers: no clock, no
// randomness, so a replay makes the same choices.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type Wire,
  type WorkloadRequest,
} from './providers.ts';

/** One (provider, model) pair that can serve a request, with its quality-table rating. */
export interface Candidate {
  providerId: string;
  wire: Wire;
  modelId: string;
  rating: number;
  probe: Probe;
  model: ProbedModel;
}

/**
 * The per-trial call budget each provider is guaranteed: the minimum of the range the kit's shared
 * provider table publishes (program/env/server.mjs, program/scorer/score.mjs and
 * program/generator/draw.mjs all carry the same table). Planning against the minimum means the plan
 * never assumes more than the provider is guaranteed to serve.
 */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** A provider the table does not name gets a quarter of the workload: useful, never dominant. */
export function capFor(providerId: string, workloadSize: number): number {
  return MIN_BUDGET[providerId] ?? Math.max(1, Math.floor(workloadSize / 4));
}

const keyOf = (c: Candidate): string => `${c.providerId}/${c.modelId}`;

/**
 * Every free model that meets the requirements, best first: rating_milli descending, then provider id,
 * then model id, so the order is fully determined. When `jsonMode` is true the client is an OpenAI one
 * asking for JSON, which the anthropic wire cannot carry, so anthropic-wire providers are excluded.
 */
export function candidatesFor(
  requires: Requirement,
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({
        providerId: probe.provider.id,
        wire: probe.provider.wire,
        modelId: model.id,
        rating: row?.rating_milli ?? 0,
        probe,
        model,
      });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0) || (a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0));
  return out;
}

/** The up-front assignment: one planned candidate per manifest row, in manifest order. */
export interface Plan {
  /** The planned candidate for each manifest row, in order; null where no candidate exists. */
  primary: (Candidate | null)[];
  workloadSize: number;
}

/**
 * Plan the whole workload greedily, in manifest order: each request takes the best candidate whose
 * provider still has planned capacity. Weight-2 requests come first in the manifest, so they get the
 * best models. When every candidate's provider is at cap, the best candidate is planned anyway: a cap
 * is a planning guard, and the actual budget may exceed the minimum.
 */
export function planAssignment(manifest: readonly WorkloadRequest[], probes: readonly Probe[], ratings: RatingsFile | null): Plan {
  const planned = new Map<string, number>();
  const primary: (Candidate | null)[] = manifest.map((request) => {
    const jsonMode = request.api === 'openai' && request.requires.json === true;
    const chain = candidatesFor(request.requires, probes, ratings, jsonMode);
    for (const candidate of chain) {
      const used = planned.get(candidate.providerId) ?? 0;
      if (used < capFor(candidate.providerId, manifest.length)) {
        planned.set(candidate.providerId, used + 1);
        return candidate;
      }
    }
    return chain[0] ?? null;
  });
  return { primary, workloadSize: manifest.length };
}

/** What the trial has learned about the providers so far, from their answers alone. */
export class RouterState {
  /** Infer attempts made per provider, failures included: the env counts every attempt the same way. */
  calls = new Map<string, number>();
  /** Successful calls per provider, for the run's summary. */
  served = new Map<string, number>();
  /** Consecutive failures per provider; two in a row trips the circuit breaker. */
  consecutiveFailures = new Map<string, number>();
  /** Providers the circuit breaker has taken out of the pool. */
  down = new Set<string>();

  recordAttempt(providerId: string): void {
    this.calls.set(providerId, (this.calls.get(providerId) ?? 0) + 1);
  }

  recordSuccess(providerId: string): void {
    this.served.set(providerId, (this.served.get(providerId) ?? 0) + 1);
    this.consecutiveFailures.set(providerId, 0);
  }

  recordFailure(providerId: string): void {
    const failures = (this.consecutiveFailures.get(providerId) ?? 0) + 1;
    this.consecutiveFailures.set(providerId, failures);
    if (failures >= 2) this.down.add(providerId);
  }

  /** Take a provider out of the pool at once: its budget is spent or it is in an outage. */
  markDown(providerId: string): void {
    this.down.add(providerId);
  }

  totalCalls(): number {
    let total = 0;
    for (const n of this.calls.values()) total += n;
    return total;
  }
}

/**
 * The next candidate to try for one request.
 *
 * - No attempt yet: the plan's primary for this request (or, when the plan has none or its provider
 *   is down, the chain's best candidate, preferring a provider still under its planned cap — an
 *   at-cap provider's budget is most likely spent, and a call to it is most likely wasted).
 * - After a failure: the chain's best candidate not yet tried for this request and not down,
 *   preferring a provider still under its planned cap, and within 10 milli of the best rating the
 *   least-called provider that has not already refused this request (a fresh provider's first call
 *   always succeeds; a provider that just refused is the same gamble on its next call number, so a
 *   second model on it is a last resort within the band). There is no same-provider retry: a refusal
 *   is usually the trial's failure draw, and the next call number on the same provider fails nearly
 *   as often as it succeeds, while a fresh provider's first call always succeeds — so falling back
 *   spends fewer calls than retrying, for a few milli of quality the score's margin absorbs.
 * - Null when the chain is exhausted: the request cannot be served.
 */
export function chooseNext(opts: {
  chain: readonly Candidate[];
  plan: Plan;
  state: RouterState;
  requestIndex: number;
  tried: readonly Candidate[];
}): Candidate | null {
  const { chain, plan, state, requestIndex, tried } = opts;
  if (chain.length === 0) return null;

  if (tried.length === 0) {
    const primary = plan.primary[requestIndex] ?? null;
    if (primary !== null && !state.down.has(primary.providerId) && chain.some((c) => keyOf(c) === keyOf(primary))) return primary;
  }

  const triedKeys = new Set(tried.map(keyOf));
  const fresh = chain.filter((c) => !triedKeys.has(keyOf(c)) && !state.down.has(c.providerId));
  const underCap = fresh.filter((c) => (state.calls.get(c.providerId) ?? 0) < capFor(c.providerId, plan.workloadSize));
  const pool = underCap.length > 0 ? underCap : fresh;
  if (pool.length === 0) return null;
  // Within 10 milli of the pool's best rating — under 0.7% of the rating, far inside the score's
  // quality margin — prefer the least-called provider: a fresh provider's first call always succeeds,
  // while a used provider's next call may hit its failure draw. A provider that already refused this
  // request sorts after one that has not: its next call number is the same gamble, so a second model
  // on it is a last resort within the band, not the pick. Rating still decides outside the band.
  const triedProviders = new Set(tried.map((c) => c.providerId));
  const bestRating = Math.max(...pool.map((c) => c.rating));
  const band = pool.filter((c) => c.rating >= bestRating - 10);
  band.sort(
    (a, b) =>
      Number(triedProviders.has(a.providerId)) - Number(triedProviders.has(b.providerId)) ||
      (state.calls.get(a.providerId) ?? 0) - (state.calls.get(b.providerId) ?? 0) ||
      b.rating - a.rating ||
      (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0) ||
      (a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0),
  );
  return band[0];
}
