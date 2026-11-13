/**
 * Round 2 — OpenRouter first-class wiring.
 *
 * Guards the swap of the OpenRouter branch in `callAnthropicMessages()` off the
 * legacy `callOpenAICompat()` shim and onto `OpenRouterProvider`, plus the
 * model-identity fix that stopped `'anthropic/'` being stripped unconditionally.
 *
 * Every test here is behavioural: it asserts what leaves the process, not what
 * the source text looks like. A string-match test cannot tell a working adapter
 * from a dead branch.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const ORIGINAL_ENV = { ...process.env };

function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Map([['content-type', 'application/json']]) as unknown as Headers,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

const CHAT_OK = {
  id: 'gen-r2',
  model: 'anthropic/claude-sonnet-4-6',
  choices: [{ message: { content: 'round two' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 },
};

type Captured = { url: string; body: Record<string, unknown> };

function stubFetch(response: unknown = CHAT_OK): Captured[] {
  const calls: Captured[] = [];
  const mock = vi.fn(async (url: unknown, init?: unknown) => {
    const body = (init as { body?: string } | undefined)?.body;
    calls.push({
      url: String(url),
      body: body ? (JSON.parse(body) as Record<string, unknown>) : {},
    });
    // Wrap in a real Response-like object. Returning the bare payload makes
    // every failure look like a code bug when it is the harness that is wrong.
    return jsonResponse(response) as never;
  });
  vi.stubGlobal('fetch', mock as unknown as typeof fetch);
  return calls;
}

const chatCalls = (calls: Captured[]) => calls.filter((c) => c.url.includes('/chat/completions'));

beforeEach(() => {
  process.env = { ...ORIGINAL_ENV };
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
  process.env = { ...ORIGINAL_ENV };
});

describe('callAnthropicMessages — OpenRouter dispatch', () => {
  it('routes OpenRouter traffic through OpenRouterProvider, not the legacy shim', async () => {
    process.env.OPENROUTER_API_KEY = 'test-key';
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_AUTH_TOKEN;
    delete process.env.OLLAMA_API_KEY;

    const calls = stubFetch();
    const { callAnthropicMessages } = await import('../src/mcp-tools/agent-execute-core.js');

    const res = await callAnthropicMessages({
      prompt: 'ping',
      model: 'openai/gpt-4.1',
      provider: 'openrouter',
    });

    expect(res.success).toBe(true);
    expect(res.output).toBe('round two');
    expect(chatCalls(calls)).toHaveLength(1);

    // The legacy shim always injected temperature: 0.7. The provider sends one
    // only when the model declares support AND a value is configured, so its
    // absence here is evidence the new path is in use.
    const body = chatCalls(calls)[0].body;
    expect(body.messages).toEqual([{ role: 'user', content: 'ping' }]);
    expect(body.max_tokens).toBe(1024);
    expect('temperature' in body).toBe(false);
  });

  it('preserves the caller model id verbatim rather than stripping a vendor prefix', async () => {
    process.env.OPENROUTER_API_KEY = 'test-key';
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_AUTH_TOKEN;

    const calls = stubFetch();
    const { callAnthropicMessages } = await import('../src/mcp-tools/agent-execute-core.js');

    // 'anthropic/claude-sonnet-4-6' is a valid OpenRouter slug. The old code
    // stripped 'anthropic/' unconditionally, sending a bare id the OpenRouter
    // endpoint rejects.
    await callAnthropicMessages({
      prompt: 'ping',
      model: 'anthropic/claude-sonnet-4-6',
      provider: 'openrouter',
    });

    expect(chatCalls(calls)[0].body.model).toBe('anthropic/claude-sonnet-4-6');
  });

  it('maps a tier alias to its OpenRouter slug', async () => {
    process.env.OPENROUTER_API_KEY = 'test-key';
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_AUTH_TOKEN;

    const calls = stubFetch();
    const { callAnthropicMessages } = await import('../src/mcp-tools/agent-execute-core.js');

    await callAnthropicMessages({ prompt: 'ping', model: 'sonnet', provider: 'openrouter' });

    expect(chatCalls(calls)[0].body.model).toBe('anthropic/claude-sonnet-4-6');
  });

  it('surfaces a named capability error instead of a generic failure', async () => {
    process.env.OPENROUTER_API_KEY = 'test-key';
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_AUTH_TOKEN;

    stubFetch();
    const { callAnthropicMessages } = await import('../src/mcp-tools/agent-execute-core.js');

    // Unknown model + tools = unverified capability. The error text must name
    // the fix, not just report failure.
    const res = await callAnthropicMessages({
      prompt: 'ping',
      model: 'nobody/never-heard-of-this',
      provider: 'openrouter',
      tools: [
        { type: 'function', function: { name: 'noop', parameters: { type: 'object', properties: {} } } },
      ],
    });

    expect(res.success).toBe(false);
    expect(res.error).toMatch(/capabilit|unverified|declare/i);
  });

  it('keeps the direct Anthropic path byte-identical when no OpenRouter key is set', async () => {
    process.env.ANTHROPIC_API_KEY = 'sk-ant-test';
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.OLLAMA_API_KEY;

    const calls = stubFetch({
      id: 'msg_1',
      model: 'claude-sonnet-4-6',
      content: [{ type: 'text', text: 'anthropic path' }],
      stop_reason: 'end_turn',
      usage: { input_tokens: 5, output_tokens: 2 },
    });
    const { callAnthropicMessages } = await import('../src/mcp-tools/agent-execute-core.js');

    const res = await callAnthropicMessages({
      prompt: 'ping',
      model: 'claude-sonnet-4-6',
      systemPrompt: 'be brief',
    });

    expect(res.success).toBe(true);
    expect(res.output).toBe('anthropic path');

    // Must still hit the native Messages endpoint, not the OpenAI-compat one.
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toContain('/v1/messages');
    expect(calls[0].url).not.toContain('chat/completions');
    // Cache breakpoint on the system block is the byte-level contract.
    expect(calls[0].body.system).toEqual([
      { type: 'text', text: 'be brief', cache_control: { type: 'ephemeral' } },
    ]);
  });
});