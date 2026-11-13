/**
 * Round 3 — N-2 and N-3.
 *
 * N-2: the ADR-149 fallback router decided whether to retry by regex-testing
 * the flattened error STRING for 429/5xx/timeout. Two ways that is wrong:
 *   - a non-retryable error whose message merely contains "503" triggers a
 *     model swap (against D7: never silently swap on a permanent fault);
 *   - a real abort/timeout carries no status and no "timeout" token, so it is
 *     NOT retried even though it is the most retryable failure there is.
 * The provider already classifies every failure with `retryable` and
 * `statusCode`. The fix carries those to the router instead of re-parsing text.
 *
 * N-3: the retry call omitted `provider`, so an agent pinned to OpenRouter
 * could have its retry silently land on a different provider.
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

describe('round 3 — retry classification and provider pinning', () => {
  let dir: string;
  let fetchMock: ReturnType<typeof vi.fn>;

  const agentsFile = () => join(dir, '.claude-flow', 'agents', 'store.json');

  beforeEach(() => {
    vi.clearAllMocks();
    dir = mkdtempSync(join(tmpdir(), 'ruflo-retry-class-'));
    for (const key of [
      'ANTHROPIC_API_KEY',
      'ANTHROPIC_AUTH_TOKEN',
      'ANTHROPIC_BASE_URL',
      'OLLAMA_API_KEY',
      'OLLAMA_BASE_URL',
      'RUFLO_PROVIDER',
      'OPENROUTER_BASE_URL',
      'CLAUDE_FLOW_ROUTER_TRAJECTORY',
      'CLAUDE_FLOW_RUN_TRANSCRIPTS',
      'CLAUDE_FLOW_ROUTER_FALLBACK_MAX_RETRIES',
    ]) {
      vi.stubEnv(key, '');
    }
    vi.stubEnv('OPENROUTER_API_KEY', 'test');
    vi.stubEnv('CLAUDE_FLOW_CWD', dir);

    const agents = join(dir, '.claude-flow', 'agents');
    mkdirSync(agents, { recursive: true });
    writeFileSync(
      agentsFile(),
      JSON.stringify({
        version: '3.0.0',
        agents: {
          test: {
            agentId: 'test',
            agentType: 'coder',
            status: 'idle',
            health: 100,
            taskCount: 0,
            config: {},
            model: 'sonnet',
            modelId: 'openai/first',
          },
        },
      }),
    );

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

  function stubFetchSequence(bodies: Array<{ status: number; body: string }>) {
    let i = 0;
    fetchMock = vi.fn().mockImplementation(async () => {
      const next = bodies[Math.min(i, bodies.length - 1)];
      i += 1;
      return {
        ok: next.status >= 200 && next.status < 300,
        status: next.status,
        headers: new Map([['content-type', 'application/json']]) as unknown as Headers,
        text: async () => next.body,
        json: async () => JSON.parse(next.body),
      };
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  it('does NOT swap models on a permanent fault whose message happens to contain 503', async () => {
    // 400 is a permanent client fault. The provider marks it retryable=false.
    // The old regex only looked at the text, so a body mentioning "503" (as
    // real upstream error pages do when describing a failed gateway) caused a
    // pointless model swap.
    const fetchSpy = stubFetchSequence([
      { status: 400, body: JSON.stringify({ error: { message: 'bad request: upstream 503 was reported earlier' } }) },
    ]);

    const result = await executeAgentTask({ agentId: 'test', prompt: 'test' });

    expect(result.success).toBe(false);
    // Exactly one request: no retry for a permanent fault.
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(mocks.next).not.toHaveBeenCalled();
  });

  it('DOES retry a real abort/timeout even though the text says neither 429 nor 5xx', async () => {
    // A caller deadline firing surfaces as AbortError -> "This operation was
    // aborted". No status, no "timeout" token. That is the single most
    // retryable failure there is, and the old regex skipped it.
    stubFetchSequence([
      { status: 200, body: JSON.stringify({ message: 'should not be used' }) },
    ]);
    const aborting = vi.fn().mockImplementation(async () => {
      const err = new Error('This operation was aborted');
      err.name = 'AbortError';
      throw err;
    });
    vi.stubGlobal('fetch', aborting);

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    // The classifier must have been consulted: an abort is retryable, so the
    // router runs its alternative-model lookup.
    expect(mocks.next).toHaveBeenCalled();
  });

  it('pins the provider on the ADR-149 retry call (N-3)', async () => {
    // The pin has to be the ONLY reason the retry reaches OpenRouter for this
    // test to mean anything. So an Anthropic key is present as well: without
    // the explicit pin, callAnthropicMessages resolves to the Anthropic path.
    // First attempt carries the pin (OpenRouter). The retry used to drop it,
    // silently landing on Anthropic.
    vi.stubEnv('ANTHROPIC_API_KEY', 'test-anthropic');

    const urls: string[] = [];
    const seq = [
      { status: 503, body: JSON.stringify({ error: { message: 'unavailable' } }) },
      { status: 200, body: JSON.stringify({
        id: 'gen-retry',
        model: 'anthropic/claude-sonnet-4-6',
        choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }) },
    ];
    let i = 0;
    const mock = vi.fn().mockImplementation(async (url: unknown) => {
      urls.push(String(url));
      const next = seq[Math.min(i, seq.length - 1)];
      i += 1;
      return {
        ok: next.status >= 200 && next.status < 300,
        status: next.status,
        headers: new Map([['content-type', 'application/json']]) as unknown as Headers,
        text: async () => next.body,
        json: async () => JSON.parse(next.body),
      };
    });
    vi.stubGlobal('fetch', mock);

    writeFileSync(
      agentsFile(),
      JSON.stringify({
        version: '3.0.0',
        agents: {
          test: {
            agentId: 'test',
            agentType: 'coder',
            status: 'idle',
            health: 100,
            taskCount: 0,
            config: {},
            model: 'sonnet',
            modelId: 'openai/first',
            provider: 'openrouter',
          },
        },
      }),
    );

    await executeAgentTask({ agentId: 'test', prompt: 'test' });

    // A retry really happened, and every attempt stayed on OpenRouter.
    expect(urls.length).toBeGreaterThanOrEqual(2);
    for (const u of urls) {
      expect(u).not.toContain('api.anthropic.com');
    }
  });
});
