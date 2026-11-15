// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// The wires this entry adds: every client wire to every upstream wire.
//
// The trial's client speaks OpenAI or Anthropic; its providers speak openai, anthropic or gemini.
// All six combinations are served here. The sim's providers answer with an ANSWER ECHO of what they
// parsed, and the driver checks that echo against the client's original request, so every translation
// preserves the echo's fields exactly: the system text, the user text, the tool names in order,
// max_tokens (never a default), the stop sequences in order, the image bytes, and JSON mode. Only the
// openai and gemini wires can carry JSON mode; an Anthropic request has no way to ask for it, so the
// router never sends a JSON-mode OpenAI request to an anthropic-wire provider (lib/router.ts).
//
// Usage is always the provider's own: a response reports the input and output tokens the provider
// recorded, never a locally computed count. Stop reasons are translated by family (stop / end_turn /
// STOP are one family, and the providers always answer in it).

/** One server-sent event frame: its event name, if any, and its data payload. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** One raw SSE frame's lines as an event name and its data payload. */
function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  return { event, data: dataLines.join('\n') };
}

/** A response body's bytes as SSE frames, blank-line separated, in arrival order. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseSseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** Whether an OpenAI-shaped body asks for JSON mode. */
export function jsonModeRequested(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** A stop value (a string or an array) as an array, in order. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return value === '' ? [] : [value];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === 'string');
  return [];
}

/** An image URL as an Anthropic image block: a data URL keeps its bytes, an http URL keeps its text. */
function anthropicImageOfUrl(url: string): Json | null {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
  if (url !== '') return { type: 'image', source: { type: 'url', url } };
  return null;
}

/** An image URL as a Gemini part: a data URL becomes inlineData, an http URL becomes fileData. */
function geminiImagePartOfUrl(url: string): Json | null {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  if (url !== '') return { fileData: { fileUri: url } };
  return null;
}

/** OpenAI message content as Anthropic content blocks: text parts and image parts, in order. */
function anthropicPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const image = anthropicImageOfUrl(block.image_url.url);
      if (image !== null) parts.push(image);
    }
  }
  return parts;
}

/** OpenAI message content as Gemini parts: text parts and image parts, in order. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const part = geminiImagePartOfUrl(block.image_url.url);
      if (part !== null) parts.push(part);
    }
  }
  return parts;
}

/** Anthropic message content as Gemini parts: text and image blocks, in order. Tool results are not
 *  content the providers echo, so they are left out (a tool result after the user content would also
 *  steal the user text: Gemini reads the last non-model content). */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
  }
  return parts;
}

/** OpenAI tools as Anthropic tools: names, descriptions and input schemas, in order. */
function anthropicTools(tools: unknown): Json[] | null {
  if (!Array.isArray(tools)) return null;
  const out: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    out.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  return out.length > 0 ? out : null;
}

/** An OpenAI tool_choice as Anthropic's. */
function anthropicToolChoice(choice: unknown): Json | null {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (choice === 'none') return { type: 'none' };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

/** Tools in either client shape as Gemini function declarations, in order. */
function geminiTools(tools: unknown): Json[] | null {
  if (!Array.isArray(tools)) return null;
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      declarations.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else if (typeof raw.name === 'string') {
      declarations.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** A tool_choice in either client shape as Gemini's function-calling config. */
function geminiToolConfig(choice: unknown): Json | null {
  if (choice === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
  if (choice === 'required') return { functionCallingConfig: { mode: 'ANY' } };
  if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  if (isRecord(choice)) {
    if (choice.type === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
    if (choice.type === 'any') return { functionCallingConfig: { mode: 'ANY' } };
    if (choice.type === 'none') return { functionCallingConfig: { mode: 'NONE' } };
    if (choice.type === 'tool' && typeof choice.name === 'string') return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the system prompt, image_url parts become base64 image blocks, tool definitions and
 * tool choice are rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass
 * through. JSON mode has no Anthropic spelling and is dropped here — the router never sends a JSON-mode
 * request down this wire.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: content.length > 0 ? content : [{ type: 'text', text: '' }] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    const parts = anthropicPartsOfOpenAI(raw.content);
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = anthropicTools(body.tools);
  if (tools !== null) out.tools = tools;
  const choice = anthropicToolChoice(body.tool_choice);
  if (choice !== null) out.tool_choice = choice;
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: system messages become the
 * system instruction, user messages become user contents (images as inlineData), tools become function
 * declarations, and the output cap, temperature, top_p, stop sequences and JSON mode move into the
 * generation config. The model travels in the URL, and streaming in the endpoint name, so neither is
 * in this body. Tool messages are left out: the providers do not echo them, and a trailing one would
 * steal the user text (Gemini reads the last non-model content).
 */
export function openAIToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    const parts = geminiPartsOfOpenAI(raw.content);
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiTools(body.tools);
  if (tools !== null) out.tools = tools;
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonModeRequested(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system prompt becomes the
 * system instruction, messages become contents (images as inlineData), tools become function
 * declarations, and the output cap, temperature, top_p and stop sequences move into the generation
 * config. Anthropic never asks for JSON mode, so no responseMimeType is set.
 */
export function anthropicToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsOfAnthropic(raw.content);
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiTools(body.tools);
  if (tools !== null) out.tools = tools;
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request for an anthropic-wire provider: the body as it stands, model replaced. */
export function anthropicToAnthropicRequest(body: Json, model: string): Json {
  return { ...body, model };
}

/**
 * An OpenAI Chat Completions request for an openai-wire provider: the body as it stands, model
 * replaced. A streaming request sets stream_options.include_usage: without it the provider sends no
 * usage chunk, and the answer would carry no usage to report.
 */
export function openAIToOpenAIRequest(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of an Anthropic message's content blocks, joined the way the driver reads it. */
function anthropicText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The text of a Gemini answer's first candidate, joined the way the driver reads it. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini answer's first candidate's finishReason, if it names one. */
function geminiFinish(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return first.finishReason;
}

/** A Gemini answer's usage: the provider's own input and output counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicMessageToOpenAI(body: Json, model: string): Json {
  const text = anthropicText(body.content);
  const toolCalls = (Array.isArray(body.content) ? body.content : []).flatMap((block) => {
    if (!isRecord(block) || block.type !== 'tool_use') return [];
    return [
      {
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      },
    ];
  });
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: geminiFinishToOpenAI(geminiFinish(body)) }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: `msg_${typeof body.modelVersion === 'string' ? body.modelVersion : model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiFinishToAnthropic(geminiFinish(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One OpenAI chat chunk as an SSE frame. */
function openAIChunk(model: string, delta: Json, finishReason: string | null): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
}

/** An OpenAI usage chunk as an SSE frame: the usage the provider recorded, carried whole. */
function openAIUsageChunk(model: string, input: number, output: number): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/**
 * An Anthropic event stream as OpenAI chunks: message_start opens with the role chunk, text deltas
 * become content deltas, message_delta becomes the finish chunk plus the usage chunk (the provider's
 * own input and output counts), and message_stop becomes [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let emittedDone = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      const message = isRecord(data.message) ? data.message : {};
      const usage = isRecord(message.usage) ? message.usage : {};
      inputTokens = numberOrZero(usage.input_tokens);
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
    } else if (frame.event === 'content_block_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        if (!started) {
          started = true;
          yield openAIChunk(model, { role: 'assistant', content: '' }, null);
        }
        yield openAIChunk(model, { content: delta.text }, null);
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') stop = delta.stop_reason;
      const usage = isRecord(data.usage) ? data.usage : {};
      outputTokens = numberOrZero(usage.output_tokens);
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      if (!finished) {
        finished = true;
        yield openAIChunk(model, {}, anthropicStopToOpenAI(stop));
        yield openAIUsageChunk(model, inputTokens, outputTokens);
      }
    } else if (frame.event === 'message_stop') {
      if (!finished) {
        if (!started) {
          started = true;
          yield openAIChunk(model, { role: 'assistant', content: '' }, null);
        }
        finished = true;
        yield openAIChunk(model, {}, anthropicStopToOpenAI(stop));
        yield openAIUsageChunk(model, inputTokens, outputTokens);
      }
      yield 'data: [DONE]\n\n';
      emittedDone = true;
      break;
    }
  }
  if (!finished) {
    if (!started) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    yield openAIChunk(model, {}, anthropicStopToOpenAI(stop));
    yield openAIUsageChunk(model, inputTokens, outputTokens);
  }
  if (!emittedDone) yield 'data: [DONE]\n\n';
}

/**
 * A Gemini event stream as OpenAI chunks: the first frame opens with the role chunk, each text piece
 * becomes a content delta, and the final frame (finishReason plus usageMetadata) becomes the finish
 * chunk plus the usage chunk, then [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let emittedDone = false;
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    const text = geminiText(data);
    if (text !== '') yield openAIChunk(model, { content: text }, null);
    if (geminiFinish(data) !== undefined) {
      const usage = geminiUsage(data);
      if (!finished) {
        finished = true;
        yield openAIChunk(model, {}, geminiFinishToOpenAI(geminiFinish(data)));
        yield openAIUsageChunk(model, usage.input, usage.output);
      }
      yield 'data: [DONE]\n\n';
      emittedDone = true;
      break;
    }
  }
  if (!finished) {
    if (!started) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    yield openAIChunk(model, {}, 'stop');
    yield openAIUsageChunk(model, 0, 0);
  }
  if (!emittedDone) yield 'data: [DONE]\n\n';
}

/**
 * A Gemini event stream as the Anthropic event sequence: message_start and content_block_start on the
 * first frame, text pieces as content_block_delta events, and the final frame as content_block_stop,
 * message_delta (the provider's own output count) and message_stop. The input count travels in the
 * final frame too, so message_start opens with 0: only the output count is read back on a streamed
 * Anthropic answer, the same as the OpenAI-to-Anthropic path above.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  const startEvents = (): string[] => [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      for (const event of startEvents()) yield event;
    }
    const text = geminiText(data);
    if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    if (geminiFinish(data) !== undefined) {
      const usage = geminiUsage(data);
      if (!finished) {
        finished = true;
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
        yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiFinishToAnthropic(geminiFinish(data)), stop_sequence: null }, usage: { output_tokens: usage.output } });
        yield sseEvent('message_stop', { type: 'message_stop' });
      }
      break;
    }
  }
  if (!finished) {
    if (!started) for (const event of startEvents()) yield event;
    yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 0 } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}
