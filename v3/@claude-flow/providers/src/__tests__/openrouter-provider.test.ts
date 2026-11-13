/**
 * OpenRouter provider — capability-gated wire behaviour.
 *
 * These tests mock global fetch and assert on the actual outbound request
 * body. They exist to lock in the behaviours that the old OpenAI-compat shim
 * got wrong: dropped tools, dropped structured output, flattened cache
 * breakpoints, and unconditional temperature injection.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OpenRouterProvider } from '../openrouter-provider.js';
import type { BaseProviderOptions } from '../base-provider.js';
import type { LLMProviderConfig } from '../types.js';

function makeProvider(overrides: Partial<LLMProviderConfig> = {}): OpenRouterProvider {
  const config: LLMProviderConfig = {
    provider: 'openrouter',
    apiKey: 'test-key-not-real',
    model: 'openai/gpt-4.1',
    ...overrides,
  } as LLMProviderConfig;

  const options: BaseProviderOptions = { config };
  const provider = new OpenRouterProvider(options);
  // doInitialize is protected; initialize() is the public path.
  return provider;
}

function stubFetch(responseBody: unknown = {}) {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const mock = vi.fn(async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return {
      ok: true,
      status: 200,
      headers: new Map() as unknown as Headers,
      json: async () => responseBody,
      text: async () => JSON.stringify(responseBody),
      body: null,
    } as unknown as Response;
  });
  vi.stubGlobal('fetch', mock);
  return { mock, calls };
}

/**
 * Chat-completion calls only. initialize() performs a /v1/models health check
 * through the same fetch, so raw call indices would be off by one.
 */
function chatCalls(calls: Array<{ url: string; init: RequestInit }>) {
  return calls.filter((c) => c.url.includes('/chat/completions'));
}

function bodyOf(calls: Array<{ url: string; init: RequestInit }>, index = 0) {
  const chat = chatCalls(calls);
  return JSON.parse(String(chat[index].init.body));
}

const BASE_RESPONSE = {
  id: 'gen-test',
  model: 'openai/gpt-4.1',
  choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
};

describe('OpenRouterProvider — request shape', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends tools and tool_choice when the model declares tool support', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'list files' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'list_files',
            description: 'List files in a directory',
            parameters: { type: 'object', properties: {} },
          },
        },
      ],
      toolChoice: 'auto',
    });

    const body = bodyOf(calls);
    expect(body.tools).toHaveLength(1);
    expect(body.tools[0].function.name).toBe('list_files');
    expect(body.tool_choice).toBe('auto');
  });

  it('sends response_format json_schema when a schema is supplied', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'give me json' }],
      providerOptions: {
        responseSchema: {
          name: 'answer',
          schema: { type: 'object', properties: { value: { type: 'string' } } },
        },
      },
    });

    const body = bodyOf(calls);
    expect(body.response_format.type).toBe('json_schema');
    expect(body.response_format.json_schema.name).toBe('answer');
  });

  it('uses multipart cache_control for Anthropic models on OpenRouter', async () => {
    const { calls } = stubFetch({ ...BASE_RESPONSE, model: 'anthropic/claude-sonnet-4-6' });
    const provider = makeProvider({ model: 'anthropic/claude-sonnet-4-6' });
    await provider.initialize();

    await provider.complete({
      messages: [
        { role: 'system', content: 'Stable instructions' },
        { role: 'user', content: 'hi' },
      ],
    });

    const body = bodyOf(calls);
    const systemMsg = body.messages.find((m: { role: string }) => m.role === 'system');
    expect(Array.isArray(systemMsg.content)).toBe(true);
    expect(systemMsg.content[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('keeps the system prompt as a plain string for implicit-prefix models', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    await provider.complete({
      messages: [
        { role: 'system', content: 'Stable instructions' },
        { role: 'user', content: 'hi' },
      ],
    });

    const body = bodyOf(calls);
    const systemMsg = body.messages.find((m: { role: string }) => m.role === 'system');
    expect(typeof systemMsg.content).toBe('string');
  });

  it('omits a provider-level temperature default for a model that rejects sampling params', async () => {
    const { calls } = stubFetch({ ...BASE_RESPONSE, model: 'anthropic/claude-opus-4-8' });
    // A provider-level default IS present. The gate must stop it reaching the
    // wire, otherwise an app-wide temperature setting makes every rejecting
    // model unusable. (Without a config value this test is tautological.)
    const provider = makeProvider({ model: 'anthropic/claude-opus-4-8', temperature: 0.7 });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
    });

    const body = bodyOf(calls);
    expect('temperature' in body).toBe(false);
  });

  it('DOES send the provider-level temperature default to a model that accepts it (positive control)', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1', temperature: 0.7 });
    await provider.initialize();

    await provider.complete({ messages: [{ role: 'user', content: 'hi' }] });

    expect(bodyOf(calls).temperature).toBe(0.7);
  });

  it('throws when the caller explicitly passes temperature to a rejecting model, before any network call', async () => {
    const { calls } = stubFetch({ ...BASE_RESPONSE, model: 'anthropic/claude-opus-4-8' });
    const provider = makeProvider({ model: 'anthropic/claude-opus-4-8' });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.3,
      })
    ).rejects.toMatchObject({ code: 'ERR_PARAM_SAMPLING_REJECTED' });

    // Prove it failed pre-flight: no chat completion ever left the process.
    expect(chatCalls(calls)).toHaveLength(0);
  });

  it('throws for an explicit top_p on a rejecting model', async () => {
    stubFetch({ ...BASE_RESPONSE, model: 'anthropic/claude-opus-4-8' });
    const provider = makeProvider({ model: 'anthropic/claude-opus-4-8' });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [{ role: 'user', content: 'hi' }],
        topP: 0.9,
      })
    ).rejects.toMatchObject({ code: 'ERR_PARAM_SAMPLING_REJECTED' });
  });

  it('throws for an explicit top_k on a rejecting model', async () => {
    stubFetch({ ...BASE_RESPONSE, model: 'anthropic/claude-opus-4-8' });
    const provider = makeProvider({ model: 'anthropic/claude-opus-4-8' });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [{ role: 'user', content: 'hi' }],
        topK: 40,
      })
    ).rejects.toMatchObject({ code: 'ERR_PARAM_SAMPLING_REJECTED' });
  });

  it('sets provider.require_parameters only when advanced parameters are present', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    // Plain text call — no routing constraint.
    await provider.complete({ messages: [{ role: 'user', content: 'hi' }] });
    expect(bodyOf(calls, 0).provider).toBeUndefined();

    // Tool call — routing constraint required (decision D2).
    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      tools: [
        {
          type: 'function',
          function: { name: 'noop', parameters: { type: 'object', properties: {} } },
        },
      ],
    });
    expect(bodyOf(calls, 1).provider).toEqual({ require_parameters: true });

    // Structured output — routing constraint required.
    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      providerOptions: {
        responseSchema: { name: 'x', schema: { type: 'object', properties: {} } },
      },
    });
    expect(bodyOf(calls, 2).provider).toEqual({ require_parameters: true });
  });

  it('does not emit an empty reasoning object, nor a routing constraint for it', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      providerOptions: { reasoning: {} },
    });

    const body = bodyOf(calls);
    expect(body.reasoning).toBeUndefined();
    expect(body.provider).toBeUndefined();
  });

  it('sends reasoning as max_tokens for a max_tokens-mode model', async () => {
    const { calls } = stubFetch({ ...BASE_RESPONSE, model: 'google/gemini-2.5-pro' });
    const provider = makeProvider({ model: 'google/gemini-2.5-pro' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      providerOptions: { reasoning: { maxTokens: 2000 } },
    });

    // toStrictEqual, not toEqual: toEqual ignores undefined-valued keys, which
    // would let a "send both shapes" regression slip through.
    expect(bodyOf(calls).reasoning).toStrictEqual({ max_tokens: 2000 });
  });

  it('sends reasoning as effort for an effort-mode model, never both shapes', async () => {
    const { calls } = stubFetch({ ...BASE_RESPONSE, model: 'openai/o3-mini' });
    const provider = makeProvider({ model: 'openai/o3-mini' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      providerOptions: { reasoning: { effort: 'high' } },
    });

    expect(bodyOf(calls).reasoning).toStrictEqual({ effort: 'high' });
  });

  it('sends ONLY the declared reasoning shape when the caller supplies both', async () => {
    const { calls } = stubFetch({ ...BASE_RESPONSE, model: 'openai/o3-mini' });
    const provider = makeProvider({ model: 'openai/o3-mini' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      providerOptions: { reasoning: { effort: 'high', maxTokens: 2000 } },
    });

    // o3-mini declares effort-mode. maxTokens must NOT reach the wire. This is
    // the case JSON.stringify cannot mask: both values are defined numbers, so
    // a "send both" regression changes the actual bytes.
    expect(bodyOf(calls).reasoning).toStrictEqual({ effort: 'high' });
  });

  it('rejects tools on a verified model whose profile declares no tool support', async () => {
    stubFetch(BASE_RESPONSE);
    const provider = new OpenRouterProvider({
      config: {
        provider: 'openrouter',
        apiKey: '***',
        model: 'vendor/no-tools-model',
        providerOptions: {
          modelCapabilities: { 'vendor/no-tools-model': { tools: 'unsupported' } },
        },
      } as unknown as LLMProviderConfig,
    });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [{ role: 'user', content: 'hi' }],
        tools: [
          {
            type: 'function',
            function: { name: 'noop', parameters: { type: 'object', properties: {} } },
          },
        ],
      })
    ).rejects.toMatchObject({ code: 'ERR_CAPABILITY_TOOLS_UNSUPPORTED' });
  });

  it('gives an UNVERIFIED model the actionable error, not a generic capability error', async () => {
    stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'totally/unknown-model-xyz' });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [{ role: 'user', content: 'hi' }],
        tools: [
          {
            type: 'function',
            function: { name: 'noop', parameters: { type: 'object', properties: {} } },
          },
        ],
      })
    ).rejects.toMatchObject({ code: 'ERR_MODEL_CAPABILITY_UNVERIFIED' });
  });

  it('allows a plain text call to an unverified model', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'totally/unknown-model-xyz' });
    await provider.initialize();

    const res = await provider.complete({ messages: [{ role: 'user', content: 'hi' }] });
    expect(res.content).toBe('ok');
    expect(chatCalls(calls)).toHaveLength(1);
  });

  it('rejects an image part on a model that declares no vision', async () => {
    stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'deepseek/deepseek-chat' });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'what is this' },
              { type: 'image', imageBase64: 'AAAA' },
            ],
          },
        ],
      })
    ).rejects.toMatchObject({ code: 'ERR_CAPABILITY_VISION_UNSUPPORTED' });
  });

  it('rejects tool_choice supplied without tools instead of dropping it', async () => {
    stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    await expect(
      provider.complete({
        messages: [{ role: 'user', content: 'hi' }],
        toolChoice: 'auto',
      })
    ).rejects.toMatchObject({ code: 'ERR_PARAM_SAMPLING_REJECTED' });
  });

  it('preserves a caller-supplied tool strict marker', async () => {
    const { calls } = stubFetch(BASE_RESPONSE);
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    await provider.complete({
      messages: [{ role: 'user', content: 'hi' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'noop',
            parameters: { type: 'object', properties: {} },
            strict: true,
          },
        } as never,
      ],
    });

    expect(bodyOf(calls).tools[0].function.strict).toBe(true);
  });

  it('normalizes tool_calls from the response', async () => {
    stubFetch({
      id: 'gen-1',
      model: 'openai/gpt-4.1',
      choices: [
        {
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              { id: 'call_1', type: 'function', function: { name: 'list_files', arguments: '{}' } },
            ],
          },
          finish_reason: 'tool_calls',
        },
      ],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    });
    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    const res = await provider.complete({
      messages: [{ role: 'user', content: 'go' }],
      tools: [
        {
          type: 'function',
          function: { name: 'list_files', parameters: { type: 'object', properties: {} } },
        },
      ],
    });

    expect(res.toolCalls).toHaveLength(1);
    expect(res.toolCalls?.[0].function.name).toBe('list_files');
    expect(res.finishReason).toBe('tool_calls');
  });
  it('yields content deltas from an OpenAI-compatible SSE stream', async () => {
    const sse = [
      'data: {"choices":[{"delta":{"content":"Hel"}}]}',
      '',
      'data: {"choices":[{"delta":{"content":"lo"}}]}',
      '',
      'data: [DONE]',
      '',
    ].join('\n');

    const encoder = new TextEncoder();
    const mock = vi.fn(async () => ({
      ok: true,
      status: 200,
      headers: new Map() as unknown as Headers,
      json: async () => ({}),
      text: async () => '',
      body: {
        getReader: () => {
          let sent = false;
          return {
            read: async () => {
              if (sent) return { done: true, value: undefined };
              sent = true;
              return { done: false, value: encoder.encode(sse) };
            },
          };
        },
      },
    })) as unknown as typeof fetch;
    vi.stubGlobal('fetch', mock);

    const provider = makeProvider({ model: 'openai/gpt-4.1' });
    await provider.initialize();

    let text = '';
    let sawDone = false;
    for await (const ev of provider.streamComplete({
      messages: [{ role: 'user', content: 'hi' }],
    })) {
      if (ev.type === 'content') text += ev.delta?.content ?? '';
      if (ev.type === 'done') sawDone = true;
    }

    // Reading `message` instead of `delta` produced zero content events.
    expect(text).toBe('Hello');
    expect(sawDone).toBe(true);
  });

});

describe('ProviderManager — openrouter factory wiring', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('creates an OpenRouterProvider instead of throwing "Unknown provider"', async () => {
    stubFetch({});
    const { ProviderManager } = await import('../provider-manager.js');

    const manager = new ProviderManager({
      providers: [
        {
          provider: 'openrouter',
          apiKey: '***',
          model: 'openai/gpt-4.1',
        },
      ],
    });

    // This is the headline bug: createProvider() threw for 'openrouter'
    // even though the LLMProvider type union lists it.
    await expect(manager.initialize()).resolves.toBeUndefined();

    const provider = manager.getProvider('openrouter');
    expect(provider).toBeDefined();
    expect(provider?.name).toBe('openrouter');
  });
});
