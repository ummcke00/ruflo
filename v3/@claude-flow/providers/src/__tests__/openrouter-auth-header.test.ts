/**
 * The Authorization header must carry the configured key.
 *
 * Found by a live run against the real OpenRouter API, which returned
 * 401 "Missing Authentication header" with a key that curl accepted. The
 * header was in fact 11 characters long — the placeholder text left behind
 * when the key interpolation was lost. Every real call was unauthenticated.
 *
 * No mocked test caught this because the mocked ones stub fetch and never
 * inspect the headers they send.
 */
import { describe, expect, it } from 'vitest';
import { OpenRouterProvider } from '../openrouter-provider.js';

// Built at runtime so no tooling can rewrite the fixture.
const KEY = ['sk', 'or', 'v1', '0123456789abcdef0123456789abcdef01234567'].join('-');

describe('OpenRouterProvider Authorization header', () => {
  it('sends the configured key, prefixed with Bearer', async () => {
    const p = new OpenRouterProvider({
      logger: { info() {}, warn() {}, error() {}, debug() {} },
      config: { provider: 'openrouter', apiKey: KEY, model: 'google/gemini-2.5-flash' },
    } as never);
    await p.initialize();
    const auth = (p as unknown as { headers: Record<string, string> }).headers.Authorization;
    expect(auth.startsWith('Bearer ')).toBe(true);
    expect(auth).toContain(KEY);
  });
});
