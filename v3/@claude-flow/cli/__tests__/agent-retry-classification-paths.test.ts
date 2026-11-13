/**
 * Round 3 review fixes — R3-1 and R3-2.
 *
 * R3-1: the provider's catch-all in BaseProvider.transformError wrapped every
 * unrecognised error as UNKNOWN with retryable=true. Now that the router trusts
 * that flag, a CODE BUG (TypeError) or a 200 with a non-JSON body (SyntaxError,
 * e.g. a proxy HTML page) triggered a model swap — 2 requests where base sent 1.
 * The abort classification in the CLI was also dead code, because the wrapper
 * renames the error before the adapter ever inspects it.
 *
 * R3-2: the Anthropic and Ollama paths returned neither statusCode nor
 * retryable, so both bugs round 3 claims to fix were still live there — and
 * Anthropic is the default path for most users.
 *
 * Every test asserts the FETCH CALL COUNT. Asserting that the router was
 * consulted is not proof a retry happened; the earlier abort test made exactly
 * that mistake and passed on unmodified code.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { executeAgentTask } from '../src/mcp-tools/agent-execute-core.js';

const mocks = vi.hoisted(() => ({ next: vi.fn(), embed: vi.fn() }));

vi.mock('../src/ruvector/neural-router.js', () => ({ nextCostOptimalAlternative: mocks.next }));
vi.mock('../src/ruvector/task-embedder.js', () => ({ embedTaskWithCache: mocks.embed }));
vi.mock('../src/ruvector/model-router.js', () => ({
  recordModelOutcome: vi.fn(),
  recordModelOutcomeByModelId: vi.fn(),
}));

describe('round 3 review — retry classification across every provider path', () => {
  let dir: string;
  const agentsFile = () => join(dir, '.claude-flow', 'agents', 'store.json');

  function writeAgent(extra: Record<string, unknown> = {}) {
    writeFileSync(
      agentsFile(),
      JSON.stringify({
        version: '3.0.0',
        agents: {
          test: {
            agentId: 'test', agentType: 'coder', status: 'idle', health: 100,
            taskCount: 0, config: {}, model: 'sonnet', modelId: 'openai/first',
            ...extra,
          },
        },
      }),
    );
  }

  beforeEach(() => {
    vi.clearAllMocks();
    dir = mkdtempSync(join(tmpdir(), 'ruflo-retry-paths-'));
    for (const key of [
      'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
      'OLLAMA_API_KEY', 'OLLAMA_BASE_URL', 'RUFLO_PROVIDER', 'OPENROUTER_BASE_URL',
      'OPENROUTER_API_KEY', 'OPENROUTER_DEFAULT_MODEL',
      'CLAUDE_FLOW_ROUTER_TRAJECTORY', 'CLAUDE_FLOW_RUN_TRANSCRIPTS',
      'CLAUDE_FLOW_ROUTER_FALLBACK_MAX_RETRIES',
    ]) {
      vi.stubEnv(key, '');
    }
    vi.stubEnv('CLAUDE_FLOW_CWD', dir);
    const agents = join(dir, '.claude-flow', 'agents');
    mkdirSync(agents, { recursive: true });
    writeAgent();

    mocks.embed.mockResolvedValue(new Float32Array([1, 0]));
    mocks.next.mockImplementation(async (_e: unknown, excluded: string[]) => ({
      modelId: `openai/retry-${excluded.length}`,
      tier: 'haiku',
    }));
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    rmSync(dir, { recursive: true, force: true });
  });

  function respond(status: number, body: string) {
    const spy = vi.fn().mockImplementation(async () => ({
      ok: status >= 200 && status < 300,
      status,
      headers: new Map([['content-type', 'application/json']]) as unknown as Headers,
      text: async () => body,
      json: async () => JSON.parse(body),
    }));
    vi.stubGlobal('fetch', spy);
    return spy;
  }

  // ---- R3-1: OpenRouter must not retry a code bug --------------------------

  it('does NOT retry an OpenRouter code bug (TypeError that is not "fetch failed")', async () => {
    vi.stubEnv('OPENROUTER_API_KEY', 'test');
    const spy = vi.fn().mockImplementation(async () => {
      throw new TypeError('x.map is not a function');
    });
    vi.stubGlobal('fetch', spy);

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('does NOT retry an OpenRouter 200 whose body is not JSON (proxy HTML page)', async () => {
    vi.stubEnv('OPENROUTER_API_KEY', 'test');
    const spy = vi.fn().mockImplementation(async () => ({
      ok: true,
      status: 200,
      headers: new Map([['content-type', 'text/html']]) as unknown as Headers,
      text: async () => '<html><body>Gateway</body></html>',
      json: async () => { throw new SyntaxError('Unexpected token <'); },
    }));
    vi.stubGlobal('fetch', spy);

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('DOES retry an OpenRouter abort, and proves it by counting requests', async () => {
    vi.stubEnv('OPENROUTER_API_KEY', 'test');
    let n = 0;
    const spy = vi.fn().mockImplementation(async () => {
      n += 1;
      if (n === 1) {
        const err = new Error('This operation was aborted');
        err.name = 'AbortError';
        throw err;
      }
      return {
        ok: true, status: 200,
        headers: new Map([['content-type', 'application/json']]) as unknown as Headers,
        text: async () => '{}',
        json: async () => ({
          id: 'gen-retry', model: 'anthropic/claude-sonnet-4-6',
          choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      };
    });
    vi.stubGlobal('fetch', spy);

    const res = await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(2);
    expect(res.success).toBe(true);
  });

  // ---- R3-2: the Anthropic path ------------------------------------------

  it('does NOT retry an Anthropic 400 whose body mentions 503', async () => {
    vi.stubEnv('ANTHROPIC_API_KEY', 'test');
    const spy = respond(400, JSON.stringify({ error: { message: 'bad: upstream 503 reported' } }));

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('does NOT retry an Anthropic 400 that merely contains the token 500', async () => {
    // Real Anthropic wording. The old regex saw "500" and swapped models.
    vi.stubEnv('ANTHROPIC_API_KEY', 'test');
    const spy = respond(400, JSON.stringify({
      error: { message: 'prompt is too long: 210000 tokens > 200000 maximum; max_tokens 500 requested' },
    }));

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('DOES retry an Anthropic 429', async () => {
    vi.stubEnv('ANTHROPIC_API_KEY', 'test');
    const spy = respond(429, JSON.stringify({ error: { message: 'rate limited' } }));

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(2);
  });

  it('DOES retry an Anthropic abort and proves it by counting requests', async () => {
    vi.stubEnv('ANTHROPIC_API_KEY', 'test');
    let n = 0;
    const spy = vi.fn().mockImplementation(async () => {
      n += 1;
      if (n === 1) {
        const err = new Error('This operation was aborted');
        err.name = 'AbortError';
        throw err;
      }
      return {
        ok: true, status: 200,
        headers: new Map([['content-type', 'application/json']]) as unknown as Headers,
        text: async () => '{}',
        json: async () => ({
          id: 'msg_1', model: 'claude-sonnet-4-6',
          content: [{ type: 'text', text: 'ok' }],
          stop_reason: 'end_turn',
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
      };
    });
    vi.stubGlobal('fetch', spy);

    const res = await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(2);
    expect(res.success).toBe(true);
  });

  // ---- R3-2: the Ollama path --------------------------------------------

  it('does NOT retry an Ollama 400 whose body mentions 502', async () => {
    vi.stubEnv('OLLAMA_API_KEY', 'test');
    const spy = respond(400, JSON.stringify({ error: { message: 'bad request: 502 from upstream' } }));

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('DOES retry an Ollama 503', async () => {
    vi.stubEnv('OLLAMA_API_KEY', 'test');
    const spy = respond(503, JSON.stringify({ error: { message: 'unavailable' } }));

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(spy).toHaveBeenCalledTimes(2);
  });
});

/**
 * R3-1 follow-up: Node raises network failures as `TypeError: fetch failed`
 * (with the real cause on `.cause`). A blanket "TypeError is not retryable"
 * rule therefore makes the most common transient fault — a dropped connection
 * — permanently unretryable. The reviewer's wording was "any TypeError OTHER
 * THAN 'fetch failed'", and the first cut of this helper missed that.
 */
describe('classifyThrownError — fetch failures are transport faults, not code bugs', () => {
  it('retries a Node "TypeError: fetch failed"', async () => {
    const { classifyThrownError } = await import('../src/mcp-tools/agent-execute-core.js');
    const err = new TypeError('fetch failed');
    expect(classifyThrownError(err).retryable).toBe(true);
  });

  it('retries a fetch failure carrying an ECONNRESET cause', async () => {
    const { classifyThrownError } = await import('../src/mcp-tools/agent-execute-core.js');
    const err = new TypeError('fetch failed');
    (err as unknown as { cause: unknown }).cause = { code: 'ECONNRESET' };
    expect(classifyThrownError(err).retryable).toBe(true);
  });

  it('still refuses a code-bug TypeError', async () => {
    const { classifyThrownError } = await import('../src/mcp-tools/agent-execute-core.js');
    expect(classifyThrownError(new TypeError('x.map is not a function')).retryable).toBe(false);
  });

  it('refuses a SyntaxError from a non-JSON body', async () => {
    const { classifyThrownError } = await import('../src/mcp-tools/agent-execute-core.js');
    expect(classifyThrownError(new SyntaxError('Unexpected token <')).retryable).toBe(false);
  });
});
