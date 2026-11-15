// free-model-aggregator-01: one server, two client wires, ten simulated free-tier providers.
//
// For each of the twenty requests the client sends, this entry serves the best free model that can
// serve it: the highest-rated free model meeting the request's requirements, on a provider that still
// has budget, with a fallback chain when a provider refuses. The score is routing quality, not just
// uptime: quality is 60% of it, and the tie-break is the number of upstream calls, so the entry plans
// the whole workload up front and spends as few calls as it can.
//
// HOW A TRIAL RUNS (the kit's RULES.txt has the full rules)
//   The trial runs this file as `node /bot/main.ts` from the repository's root, with Node 22's
//   standard library alone (TypeScript runs directly; a relative import needs its .ts extension).
//   The brief is at the path in $MAKEFUN_PROMPT_FILE: the provider directory, the quality table
//   (every model's rating and capabilities) and the workload manifest (the twenty requests, in order).
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server:
//     1. reads the brief, probes each provider's catalog route in the provider's own wire, and builds
//        one catalog (the `--list` document and the register body, the same object);
//     2. plans the whole workload: one candidate per request, best-rated first, each provider capped
//        at the minimum of its published per-trial budget range (lib/router.ts);
//     3. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     4. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     5. serves the client's requests: GET /v1/models (in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     6. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//        A process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts.
//
// THE ROUTING (lib/router.ts)
//   - Candidates are the free models meeting a request's requirements, ranked by the quality table's
//     rating_milli. The plan assigns them greedily in manifest order under per-provider caps, so the
//     weight-2 requests (which come first) get the best models and no provider is asked for more than
//     its guaranteed budget.
//   - A refusal (402, 429, 5xx) is answered by falling back down the rating-ordered chain, preferring
//     providers still under their planned cap. A provider whose budget is spent or that is in an
//     outage is retired at once (its refusal names the cause); a schedule refusal counts toward a
//     circuit breaker that takes a provider out after two consecutive failures. There is no
//     same-provider retry: a fresh provider's first call always succeeds, so falling back spends
//     fewer calls than retrying.
//   - Exactly one successful upstream call is made per request: the attempt loop stops at the first
//     success, and a success is never followed by another call.
//
// THE WIRES (lib/wire.ts)
//   All six client/upstream combinations are served: OpenAI and Anthropic clients, each to openai,
//   anthropic and gemini providers, JSON and streaming. Every translation preserves the echo the
//   providers answer with — system text, user text, tool names in order, max_tokens, stop sequences,
//   image bytes and JSON mode — and every response reports the provider's own usage and stop family.
//   A streamed OpenAI request sets stream_options.include_usage upstream, so the provider sends the
//   usage chunk the answer needs.
//
// DETERMINISM
//   Nothing here reads the clock or rolls dice: every choice derives from the brief, the probes and
//   the providers' answers, so a replay makes the same choices and leaves the same state.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/router.ts      the rating-ranked candidates, the up-front plan, the failover policy.
//   lib/wire.ts        the client and upstream wires: request and response translation, SSE.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicMessageToOpenAI,
  anthropicModelList,
  anthropicToAnthropicRequest,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  jsonModeRequested,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  openAIToOpenAIRequest,
  type ServedModel,
} from './lib/wire.ts';
import { candidatesFor, chooseNext, planAssignment, RouterState, type Candidate, type Plan } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How many times a failed catalog probe is retried: a missing provider costs catalog checks. */
const PROBE_ATTEMPTS = 3;
/** The pause between probe retries, in milliseconds (a wait, not a value in any request). */
const PROBE_RETRY_MS = 200;
/** How long one upstream infer call may take, in milliseconds (the client allows 30 s a request). */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most attempts one request gets, across the fallback chain: one per provider, so a request is
 *  only unserved when every provider that meets it has refused. Serving a request is worth far more
 *  than the tie-break a call costs, so the cap binds only in a pathological cascade. */
const MAX_ATTEMPTS = 10;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plan: Plan = { primary: [], workloadSize: 0 };
const routerState = new RouterState();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider (retrying a failed probe) and build the one catalog and the plan. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map(async (provider) => {
      let probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
      for (let attempt = 1; !probe.ok && attempt < PROBE_ATTEMPTS; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, PROBE_RETRY_MS));
        probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
      }
      return probe;
    }),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = planAssignment(brief.workload?.manifest ?? [], probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: jsonModeRequested(body),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request for one candidate: the URL, headers and body in the provider's own wire. */
function upstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): { url: string; headers: Record<string, string>; body: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstreamBody = clientApi === 'openai' ? openAIToOpenAIRequest(body, candidate.modelId) : anthropicToOpenAI(body, candidate.modelId);
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody = clientApi === 'openai' ? openAIToAnthropicRequest(body, candidate.modelId) : anthropicToAnthropicRequest(body, candidate.modelId);
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}:${action}`, headers, body: upstreamBody };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write an SSE head, then the translated events of one upstream stream; always end the response. */
async function sendStreamEvents(res: ServerResponse, events: AsyncGenerator<string>): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    for await (const event of events) res.write(event);
  } finally {
    res.end();
  }
}

/**
 * Answer the client from one successful upstream call, translating the provider's wire into the
 * client's. A non-streaming answer is read whole and converted; a streaming one is translated event by
 * event, or piped verbatim when the wires already match.
 */
async function sendAnswer(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstream: Response,
  stream: boolean,
  res: ServerResponse,
): Promise<boolean> {
  const model = candidate.modelId;
  if (!stream) {
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      return false;
    }
    const body = isRecord(answer) ? answer : {};
    if (clientApi === 'openai') {
      if (candidate.wire === 'openai') sendJson(res, 200, body);
      else if (candidate.wire === 'anthropic') sendJson(res, 200, anthropicMessageToOpenAI(body, model));
      else sendJson(res, 200, geminiToOpenAI(body, model));
    } else {
      if (candidate.wire === 'openai') sendJson(res, 200, openAIToAnthropic(body, model));
      else if (candidate.wire === 'anthropic') sendJson(res, 200, body);
      else sendJson(res, 200, geminiToAnthropic(body, model));
    }
    return true;
  }
  if (upstream.body === null) return false;
  if (clientApi === 'openai' && candidate.wire === 'openai') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return true;
  }
  if (clientApi === 'anthropic' && candidate.wire === 'anthropic') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return true;
  }
  if (clientApi === 'openai' && candidate.wire === 'anthropic') {
    await sendStreamEvents(res, openAIEventsFromAnthropic(upstream.body, model));
    return true;
  }
  if (clientApi === 'openai' && candidate.wire === 'gemini') {
    await sendStreamEvents(res, openAIEventsFromGemini(upstream.body, model));
    return true;
  }
  if (clientApi === 'anthropic' && candidate.wire === 'openai') {
    await sendStreamEvents(res, anthropicEventsFromOpenAI(upstream.body, model));
    return true;
  }
  await sendStreamEvents(res, anthropicEventsFromGemini(upstream.body, model));
  return true;
}

/** Whether an upstream failure is worth falling back from (a refusal, not a bad request). */
function retryableStatus(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/**
 * What a provider's refusal means, from the message it answered with. The providers name the cause:
 * "{name}'s free-tier budget is spent" and "{name} is out" are terminal — every further call to that
 * provider fails — while "{name} refused this call" is the trial's failure draw and the next call
 * number may succeed. The message is read from the error envelope (all three wires carry
 * error.message); an unparseable body falls back to the raw text.
 */
function failureKind(bodyText: string): 'spent' | 'outage' | 'refused' {
  let message = bodyText;
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.message === 'string') message = parsed.error.message;
  } catch {
    // Not JSON: match the raw text.
  }
  if (message.includes('budget is spent')) return 'spent';
  if (message.includes(' is out')) return 'outage';
  return 'refused';
}

/**
 * Serve one client request: route it, call the provider, and answer in the client's wire. The attempt
 * loop walks the router's choices — the plan's primary, then the rating-ordered fallback chain — and
 * stops at the first success, so exactly one upstream call answers the request. A provider whose
 * budget is spent or that is in an outage is retired at once; a schedule refusal counts toward the
 * circuit breaker.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requestIndex = manifestIndex;
  manifestIndex += 1;
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[requestIndex];
  const requestId = entry?.id ?? `request-${requestIndex + 1}`;
  const requires = entry !== undefined ? entry.requires : requirementsFromBody(body);
  const jsonMode = clientApi === 'openai' && jsonModeRequested(body);
  const chain = candidatesFor(requires, probes, brief.ratings, jsonMode);
  if (chain.length === 0) {
    log(`${requestId}: no free model meets the requirements`);
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const stream = body.stream === true;
  const tried: Candidate[] = [];
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const candidate = chooseNext({ chain, plan, state: routerState, requestIndex, tried });
    if (candidate === null) {
      log(`${requestId}: the fallback chain is exhausted after ${tried.length} attempts`);
      return sendError(res, clientApi, 503, 'no_free_model', 'every free model that meets this request refused it');
    }
    tried.push(candidate);
    routerState.recordAttempt(candidate.providerId);
    const { url, headers, body: upstreamBody } = upstreamRequest(candidate, clientApi, body, stream);
    let upstream: Response;
    try {
      upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
    } catch (err) {
      routerState.recordFailure(candidate.providerId);
      log(`${requestId}: attempt ${attempt + 1} ${candidate.providerId}/${candidate.modelId} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      const detail = await upstream.text();
      const kind = failureKind(detail);
      if (kind === 'spent' || kind === 'outage') routerState.markDown(candidate.providerId);
      else routerState.recordFailure(candidate.providerId);
      if (!retryableStatus(upstream.status)) log(`${requestId}: attempt ${attempt + 1} ${candidate.providerId}/${candidate.modelId} HTTP ${upstream.status} (not retryable): ${detail.slice(0, 200)}`);
      else log(`${requestId}: attempt ${attempt + 1} ${candidate.providerId}/${candidate.modelId} HTTP ${upstream.status} (${kind}): ${detail.slice(0, 200)}`);
      continue;
    }
    const answered = await sendAnswer(clientApi, candidate, upstream, stream, res);
    if (!answered) {
      routerState.recordFailure(candidate.providerId);
      log(`${requestId}: attempt ${attempt + 1} ${candidate.providerId}/${candidate.modelId} answered an unreadable body`);
      continue;
    }
    routerState.recordSuccess(candidate.providerId);
    log(`${requestId}: served by ${candidate.providerId}/${candidate.modelId} (rating ${candidate.rating}, attempt ${attempt + 1})`);
    return;
  }
  log(`${requestId}: still unserved after ${MAX_ATTEMPTS} attempts`);
  sendError(res, clientApi, 503, 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** One line per provider for the run's summary, in the brief's provider order. */
function summary(): string {
  const parts: string[] = [];
  for (const probe of probes) {
    const served = routerState.served.get(probe.provider.id) ?? 0;
    const calls = routerState.calls.get(probe.provider.id) ?? 0;
    if (calls > 0) parts.push(`${probe.provider.id} ${served}/${calls}`);
  }
  return `done: ${routerState.totalCalls()} upstream calls for ${manifestIndex} requests (${parts.join(', ')})`;
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          log(summary());
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
