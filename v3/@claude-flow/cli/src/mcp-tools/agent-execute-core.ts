/**
 * Shared agent-execution core.
 *
 * Both the agent_execute MCP tool and the workflow runtime (G3) need
 * to dispatch a prompt to an agent's configured Anthropic model. This
 * module factors that path out so it's testable and reusable, and
 * keeps the wire from agent_spawn → ProviderManager (real) in one
 * place rather than duplicated.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { getProjectCwd } from './types.js';
import { appendAgentExecution, summarizeAgentExecutions, type AgentExecutionSample } from './agent-health-metrics.js';
import { configManager } from '../services/config-file-manager.js';
// Type-only: erased at compile time, so importing this module costs nothing.
// The provider VALUE is loaded lazily inside callOpenRouterProvider (see there)
// so that commands which never touch OpenRouter — and the isolated sandbox in
// __tests__/analyze-json-output.test.ts — do not need the package resolvable.
import type { LLMRequest, LLMResponse } from '@claude-flow/providers';

const STORAGE_DIR = '.claude-flow';
const AGENT_DIR = 'agents';
const AGENT_FILE = 'store.json';

type ClaudeModel = 'haiku' | 'sonnet' | 'opus' | 'opus-4.7' | 'inherit';

export interface AgentRecord {
  agentId: string;
  agentType: string;
  status: 'idle' | 'busy' | 'terminated';
  health: number;
  taskCount: number;
  config: Record<string, unknown>;
  createdAt: string;
  domain?: string;
  model?: ClaudeModel;
  modelRoutedBy?: 'explicit' | 'router' | 'codemod' | 'default' | 'hybrid';
  /**
   * ADR-149 — concrete picked model id (e.g. `openai/gpt-4.1`,
   * `inclusionai/ling-2.6-flash`). Present when the cost-optimal neural
   * router contributed to the decision; downstream `executeAgentInline`
   * uses this to dispatch via the correct provider's API instead of
   * falling back to MODEL_MAP[tier].
   */
  modelId?: string;
  /** ADR-148 phase 2 — execution provider hint. #2962 widened to include 'ollama'. */
  provider?: 'anthropic' | 'openrouter' | 'ollama';
  /** ADR-148 phase 2 — concrete OpenRouter slug when provider='openrouter'. */
  openrouterModel?: string;
  lastResult?: Record<string, unknown>;
  executionHistory?: AgentExecutionSample[];
}

interface AgentStore {
  agents: Record<string, AgentRecord>;
  version: string;
}

function getAgentDir(): string { return join(getProjectCwd(), STORAGE_DIR, AGENT_DIR); }
function getAgentPath(): string { return join(getAgentDir(), AGENT_FILE); }
function ensureAgentDir(): void {
  const dir = getAgentDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}
function loadAgentStore(): AgentStore {
  try {
    if (existsSync(getAgentPath())) return JSON.parse(readFileSync(getAgentPath(), 'utf-8'));
  } catch { /* fall through */ }
  return { agents: {}, version: '3.0.0' };
}
function saveAgentStore(store: AgentStore): void {
  ensureAgentDir();
  writeFileSync(getAgentPath(), JSON.stringify(store, null, 2), 'utf-8');
}

// #1906/#2232 — Current model ids (Claude 4.x family):
//   Opus 4.8    → claude-opus-4-8   (current, the `opus` alias)
//   Opus 4.7    → claude-opus-4-7   (prior pin, reachable via `opus-4.7`)
//   Sonnet 5    → claude-sonnet-5   (current, the `sonnet` alias)
//   Sonnet 4.6  → claude-sonnet-4-6 (prior pin, reachable via `sonnet-4.6`)
//   Haiku 4.5   → claude-haiku-4-5-20251001
// `inherit` and the various defaults below all map to Sonnet 5.
export const DEFAULT_ANTHROPIC_MODEL = 'claude-sonnet-5';
const MODEL_MAP: Record<string, string> = {
  haiku: 'claude-haiku-4-5-20251001',
  sonnet: 'claude-sonnet-5',
  'sonnet-4.6': 'claude-sonnet-4-6',
  opus: 'claude-opus-4-8',
  'opus-4.7': 'claude-opus-4-7',
  inherit: DEFAULT_ANTHROPIC_MODEL,
};

// #2357 — the adaptive-thinking family (Fable 5, Opus 4.8, Opus 4.7, Sonnet 5)
// removed the sampling parameters (temperature/top_p/top_k); the Anthropic API
// returns 400 "Extra inputs are not permitted" when any is present.
// Prefix-match so dated snapshots (e.g. claude-opus-4-8-YYYYMMDD) are
// covered. Applies only to the direct Anthropic path — the Ollama/OpenRouter
// OpenAI-compat paths accept temperature and are unchanged.
export function modelRejectsSamplingParams(model: string): boolean {
  return /^claude-(fable-5|opus-4-8|opus-4-7|sonnet-5)/.test(model);
}

export interface AnthropicCallInput {
  prompt: string;
  systemPrompt?: string;
  model?: string;          // already-resolved Anthropic model id (e.g. 'claude-sonnet-4-6')
  maxTokens?: number;
  temperature?: number;
  timeoutMs?: number;
  /**
   * #2962 — explicit provider carried from the agent record (agent.provider,
   * itself populated from a user's `--provider` flag or persisted
   * `providers configure`). When set, this outranks the env-var-only
   * inference callAnthropicMessages otherwise does — see the precedence
   * comment on that function.
   */
  provider?: 'anthropic' | 'openrouter' | 'ollama';
  /**
   * Tool definitions to advertise to the model. Only the OpenRouter path
   * consumes these today; the direct Anthropic single-turn path ignores them
   * (and therefore stays byte-identical). Passing tools to a model that does
   * not declare support raises a named capability error rather than being
   * silently dropped.
   */
  tools?: unknown[];
}

/** A single entry from the persisted `agents.providers` config array. */
interface PersistedProviderEntry {
  name: string;
  enabled?: boolean;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
}

/**
 * #2962 — read the enabled `agents.providers` entry matching `name` from
 * cwd-scoped `claude-flow.config.json` (written by `providers configure`,
 * `commands/providers.ts`). Best-effort: any failure (no config file, wrong
 * shape, etc.) returns undefined rather than throwing — this must never
 * break execution, and env vars remain a fully-supported override that
 * doesn't require this file to exist.
 */
function getPersistedProviderConfig(name: string): PersistedProviderEntry | undefined {
  try {
    const providers = configManager.get(getProjectCwd(), 'agents.providers');
    if (!Array.isArray(providers)) return undefined;
    const entry = providers.find(
      (p): p is PersistedProviderEntry =>
        !!p && typeof p === 'object' && typeof (p as PersistedProviderEntry).name === 'string' &&
        (p as PersistedProviderEntry).name.toLowerCase() === name.toLowerCase(),
    );
    if (!entry || entry.enabled === false) return undefined;
    return entry;
  } catch {
    return undefined;
  }
}

/**
 * Is this HTTP status worth retrying on a different model?
 *
 * 408/425 are transport-ish, 429 is rate limiting, 5xx is the server's fault.
 * Everything else (400/401/403/404/409/413/422, ...) is a permanent client or
 * auth fault: retrying it on another model cannot succeed and just doubles the
 * user's latency.
 */
export function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

/**
 * Classify an error thrown out of a fetch/json call.
 *
 * Aborts and timeouts are retryable. A TypeError or SyntaxError means OUR code
 * or the response shape is wrong — a proxy's HTML error page served with 200,
 * for instance — which fails identically on every model, so retrying it only
 * doubles latency. Anything unrecognised is treated as NOT retryable: the
 * caller branches on this, so a default of "retry" would turn every unknown
 * fault into a silent model swap.
 */
export function classifyThrownError(err: unknown): { retryable: boolean; statusCode?: number } {
  if (err instanceof Error) {
    if (err.name === 'AbortError' || err.name === 'TimeoutError') return { retryable: true };
    if (err.name === 'SyntaxError' || err.name === 'TypeError' || err.name === 'ReferenceError') {
      return { retryable: false };
    }
    const message = err.message ?? '';
    if (/\b(ECONNRESET|ETIMEDOUT|ECONNREFUSED|EAI_AGAIN)\b/.test(message) || /fetch failed/i.test(message)) {
      return { retryable: true };
    }
  }
  return { retryable: false };
}

/**
 * Decide whether a failed call is worth retrying on another model.
 *
 * The provider already classifies every failure: RufloProviderError carries an
 * explicit `retryable` (false for auth, capability and other permanent faults),
 * LLMProviderError carries `statusCode`. Read those.
 *
 * This used to regex-test `result.error` for 429/5xx/timeout, which is wrong in
 * both directions: a 400 body can mention "503" and trigger a pointless model
 * swap, while a real AbortError ("This operation was aborted") has no status
 * and no "timeout" token and was therefore never retried at all.
 *
 * The text check survives only as a last resort for results that predate the
 * structured fields (an older caller or a provider without classification).
 */
export function isRetryableFailure(result: {
  success?: boolean;
  error?: string;
  retryable?: boolean;
  statusCode?: number;
}): boolean {
  if (result.success) return false;
  if (typeof result.retryable === 'boolean') return result.retryable;

  const status = result.statusCode;
  if (typeof status === 'number') {
    return status === 408 || status === 425 || status === 429 || status >= 500;
  }

  return /\b(429|500|502|503|504|timeout|ECONNRESET|ETIMEDOUT)\b/i.test(result.error ?? '');
}

export interface AnthropicCallResult {
  success: boolean;
  model?: string;
  messageId?: string;
  stopReason?: string;
  output?: string;
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number };
  durationMs?: number;
  error?: string;
  /**
   * The provider's OWN verdict on whether this failure is worth retrying, and
   * the status it came with. The ADR-149 fallback router reads these instead of
   * regex-testing `error`, because text is not a reliable classifier: a 400 body
   * can mention "503", and a real AbortError carries no status at all.
   * Optional so every existing caller and test keeps compiling.
   */
  retryable?: boolean;
  statusCode?: number;
}

/**
 * Generic Anthropic Messages API call. No agent registry coupling — used
 * by agent_execute (with the agent's configured model) and by the WASM
 * agent runtime (G4) when the bundled WASM only echoes input.
 *
 * #1725 — falls back to Ollama Cloud (Tier-2, OpenAI-compat) when
 * ANTHROPIC_API_KEY is unset and OLLAMA_API_KEY is present, or when
 * RUFLO_PROVIDER=ollama is explicitly set. Response shape is normalized
 * to the Anthropic-flavored AnthropicCallResult so existing callers
 * don't need to know which provider answered.
 */
export async function callAnthropicMessages(input: AnthropicCallInput): Promise<AnthropicCallResult> {
  // #2962 — precedence: explicit per-agent flag (input.provider, forwarded
  // from agent.provider by executeAgentTask, itself populated from a
  // user's `agent spawn --provider` flag) → env vars (RUFLO_PROVIDER + the
  // *_API_KEY family — unchanged back-compat surface) → persisted
  // `agents.providers` config (`providers configure`) → the original
  // key-presence inference, kept below as the last-resort fallback.
  const explicitProvider = (input.provider || process.env.RUFLO_PROVIDER || '').toLowerCase();
  const ollamaKey = process.env.OLLAMA_API_KEY;
  const anthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN;
  // A gateway bearer token is also an Anthropic credential for provider inference.
  const anthropicKey = process.env.ANTHROPIC_API_KEY || anthropicAuthToken;
  // #2042 — OpenRouter is an OpenAI-compat endpoint that fronts dozens of
  // providers. Reporter (@ummcke00) had `providers.openrouter.apiKey` in
  // their config.yaml but agent_execute hardcoded Anthropic. Detect via
  // explicit RUFLO_PROVIDER=openrouter OR presence of OPENROUTER_API_KEY
  // when no Anthropic key is available (same precedence as the Ollama
  // branch above).
  const openrouterKey = process.env.OPENROUTER_API_KEY;
  const useOpenRouter =
    explicitProvider === 'openrouter' || (!explicitProvider && !anthropicKey && !!openrouterKey);
  // #2962 — only consult the persisted config when a candidate is actually
  // relevant (explicit choice, or no env key found anywhere), so a normal
  // ANTHROPIC_API_KEY-only setup never pays a config-file read.
  const persistedOllama =
    explicitProvider === 'ollama' || (!anthropicKey && !openrouterKey && !ollamaKey)
      ? getPersistedProviderConfig('ollama')
      : undefined;
  const persistedOpenRouter =
    explicitProvider === 'openrouter' && !openrouterKey ? getPersistedProviderConfig('openrouter') : undefined;
  const useOllama =
    explicitProvider === 'ollama' || (!explicitProvider && !anthropicKey && !openrouterKey && (!!ollamaKey || !!persistedOllama));

  if (useOpenRouter) {
    const apiKey = openrouterKey || persistedOpenRouter?.apiKey;
    if (apiKey) {
      return callOpenRouterProvider({
        ...input,
        apiKey,
        baseUrl: process.env.OPENROUTER_BASE_URL || persistedOpenRouter?.baseUrl || 'https://openrouter.ai/api',
        // #2357 Finding C: anthropic/claude-3.5-sonnet was retired Oct 2025.
        // Default to the same canonical family the rest of the resolver uses
        // (MODEL_MAP). `OPENROUTER_DEFAULT_MODEL` still wins for callers who
        // want to pin a specific OpenRouter slug.
        defaultModel: process.env.OPENROUTER_DEFAULT_MODEL || persistedOpenRouter?.model || 'anthropic/claude-sonnet-4-6',
      });
    }
    return { success: false, error: 'OpenRouter was selected but no API key is configured.' };
  }
  if (useOllama) {
    // #2962 — retire the undocumented OLLAMA_API_KEY=local sentinel
    // requirement. A self-hosted, unauthenticated Ollama daemon shouldn't
    // need a fake credential to become reachable; "self-hosted" is any
    // resolved base URL that isn't the public Ollama Cloud endpoint
    // (persisted `providers configure -e` baseUrl, or OLLAMA_BASE_URL,
    // checked in that order — matching callOllamaCompat's own precedence).
    const resolvedBaseUrl = process.env.OLLAMA_BASE_URL || persistedOllama?.baseUrl;
    const isSelfHosted = !!resolvedBaseUrl && !/^https:\/\/ollama\.com\/?$/i.test(resolvedBaseUrl);
    if (ollamaKey || persistedOllama?.apiKey || isSelfHosted) {
      return callOllamaCompat({
        ...input,
        apiKey: ollamaKey || persistedOllama?.apiKey || 'local',
        baseUrl: resolvedBaseUrl,
        model: input.model,
        // Local default for tier aliases / Anthropic ids (see
        // resolveOllamaModel): OLLAMA_DEFAULT_MODEL, then the model saved by
        // `providers configure -p ollama -m <tag>`.
        defaultModel: process.env.OLLAMA_DEFAULT_MODEL || persistedOllama?.model,
      });
    }
    return { success: false, error: 'Ollama was selected but no API key or self-hosted endpoint is configured.' };
  }
  if (!anthropicKey) {
    return {
      success: false,
      error:
        'No LLM provider configured. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN (Tier-3), OPENROUTER_API_KEY (#2042), or OLLAMA_API_KEY (Tier-2 — #1725).',
    };
  }
  const model = input.model || DEFAULT_ANTHROPIC_MODEL;
  const startedAt = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), input.timeoutMs || 60000);
  try {
    const baseUrl = (process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com').replace(/\/+$/, '');
    const res = await fetch(`${baseUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        ...(anthropicAuthToken
          ? { Authorization: `Bearer ${anthropicAuthToken}` }
          : { 'x-api-key': anthropicKey }),
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: input.maxTokens || 1024,
        // #2357 — omit temperature for models that reject sampling params
        // (Fable 5 / Opus 4.8 / Opus 4.7 → 400 "Extra inputs are not
        // permitted"); keep the 0.7 default unchanged for models that still
        // accept it (sonnet / haiku / opus ≤4.6).
        ...(modelRejectsSamplingParams(model)
          ? {}
          : { temperature: typeof input.temperature === 'number' ? input.temperature : 0.7 }),
        // #8 prompt caching (hermes-agent pattern): mark the (often large,
        // stable) system prompt as an ephemeral cache breakpoint so repeated
        // agent_execute calls with the same system prompt hit Anthropic's
        // prompt cache (~90% discount on cached input tokens, 5-min TTL).
        ...(input.systemPrompt
          ? { system: [{ type: 'text', text: input.systemPrompt, cache_control: { type: 'ephemeral' } }] }
          : {}),
        messages: [{ role: 'user', content: input.prompt }],
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => '<unreadable error body>');
      // R3-2 — the Anthropic path is the DEFAULT path for most users. It used
      // to return neither statusCode nor retryable, so it fell through to the
      // text regex and still swapped models on a 400 whose body mentioned
      // "503", or on "prompt is too long ... max_tokens 500".
      return {
        success: false,
        model,
        error: `Anthropic API error ${res.status}: ${errText.slice(0, 400)}`,
        statusCode: res.status,
        retryable: isRetryableStatus(res.status),
      };
    }
    const data = await res.json() as {
      id: string;
      model: string;
      content: Array<{ type: string; text?: string }>;
      stop_reason: string;
      usage: { input_tokens: number; output_tokens: number };
    };
    const textOut = data.content
      .filter(c => c.type === 'text' && typeof c.text === 'string')
      .map(c => c.text as string)
      .join('');
    return {
      success: true,
      model: data.model,
      messageId: data.id,
      stopReason: data.stop_reason,
      output: textOut,
      usage: {
        inputTokens: data.usage.input_tokens,
        outputTokens: data.usage.output_tokens,
        totalTokens: data.usage.input_tokens + data.usage.output_tokens,
      },
      durationMs: Date.now() - startedAt,
    };
  } catch (err) {
    // R3-2 — an abort here is the most retryable failure there is, and it was
    // never retried because it carries no status and no "timeout" token.
    const classification = classifyThrownError(err);
    return {
      success: false,
      model,
      error: err instanceof Error ? err.message : String(err),
      ...classification,
      durationMs: Date.now() - startedAt,
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ollama Cloud / OpenAI-compat provider — Tier-2 routing per ADR-026 + #1725.
 *
 * Endpoint: https://ollama.com/v1/chat/completions
 * Auth: Authorization: Bearer <OLLAMA_API_KEY>
 *
 * Translates the Anthropic-flavored input shape onto OpenAI chat-completions
 * and translates the response back so callers never see provider-specific
 * fields. Logical model names ('haiku' / 'sonnet' / 'opus' / a tier-routed
 * `claude-*` id) resolve to the configured OLLAMA_DEFAULT_MODEL / persisted
 * default; only when the endpoint is genuinely Ollama's cloud service AND no
 * default is configured does this fall back to 'gpt-oss:120b-cloud' — a
 * self-hosted/local endpoint with no default configured fails loudly instead
 * (see resolveOllamaModel). An explicit 'ollama:<model>' or bare
 * provider-native name always passes through unchanged.
 */
async function callOllamaCompat(
  input: AnthropicCallInput & { apiKey: string; baseUrl?: string; defaultModel?: string },
): Promise<AnthropicCallResult> {
  const startedAt = Date.now();
  // #2962 — input.baseUrl (resolved by the caller from persisted
  // `providers configure` config, then OLLAMA_BASE_URL) takes precedence
  // over re-reading OLLAMA_BASE_URL here, so a persisted-config-only setup
  // (no env vars) still reaches a self-hosted endpoint instead of Ollama
  // Cloud. Falls back to the original OLLAMA_BASE_URL-or-cloud behavior
  // for any direct caller that doesn't pass baseUrl.
  const base = (input.baseUrl || process.env.OLLAMA_BASE_URL || 'https://ollama.com').replace(/\/+$/, '');
  // Only a base URL actually pointed at Ollama's real cloud endpoint may
  // fall back to a `-cloud`-suffixed default model — see resolveOllamaModel.
  const isCloudEndpoint = /^https:\/\/ollama\.com\/?$/i.test(base);
  let model: string;
  try {
    model = resolveOllamaModel(input.model, input.defaultModel, isCloudEndpoint);
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : String(err),
      durationMs: Date.now() - startedAt,
    };
  }
  const url = `${base}/v1/chat/completions`;
  // Self-hosted endpoints typically don't need an Authorization header
  // (the daemon binds to 11434 with no auth by default), but Ollama Cloud
  // does. Send the bearer when the key is non-empty AND looks cloud-shaped.
  const sendAuth = !!input.apiKey && input.apiKey !== 'local';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), input.timeoutMs || 60000);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        ...(sendAuth ? { Authorization: `Bearer ${input.apiKey}` } : {}),
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: input.maxTokens || 1024,
        temperature: typeof input.temperature === 'number' ? input.temperature : 0.7,
        messages: [
          ...(input.systemPrompt
            ? [{ role: 'system' as const, content: input.systemPrompt }]
            : []),
          { role: 'user' as const, content: input.prompt },
        ],
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => '<unreadable error body>');
      // R3-2 — same gap as the Anthropic path: without a structured verdict the
      // router regex-matched the body and swapped models on permanent faults.
      return {
        success: false,
        model,
        error: `Ollama API error ${res.status} at ${url}: ${errText.slice(0, 400)}`,
        statusCode: res.status,
        retryable: isRetryableStatus(res.status),
      };
    }
    const data = (await res.json()) as {
      id?: string;
      model?: string;
      choices: Array<{
        message: { role: string; content: string };
        finish_reason?: string;
      }>;
      usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
      };
    };
    const textOut = data.choices?.[0]?.message?.content ?? '';
    const usage = data.usage ?? {};
    return {
      success: true,
      model: data.model ?? model,
      messageId: data.id ?? `ollama-${Date.now()}`,
      stopReason: data.choices?.[0]?.finish_reason ?? 'end_turn',
      output: textOut,
      usage: {
        inputTokens: usage.prompt_tokens ?? 0,
        outputTokens: usage.completion_tokens ?? 0,
        totalTokens: usage.total_tokens ?? 0,
      },
      durationMs: Date.now() - startedAt,
    };
  } catch (err) {
    const classification = classifyThrownError(err);
    return {
      success: false,
      model,
      error: err instanceof Error ? err.message : String(err),
      ...classification,
      durationMs: Date.now() - startedAt,
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * OpenRouter adapter — the first-class path.
 *
 * Routes through `OpenRouterProvider` from `@claude-flow/providers` instead of
 * hand-rolling a chat-completions fetch. That buys three things the old shim
 * could not do:
 *
 *   1. **Declared capabilities.** Tools, structured output, reasoning control,
 *      vision and audio are gated on the model's declared profile rather than
 *      guessed from its name or silently dropped.
 *   2. **Multipart prompt caching.** A system prompt becomes a content-part
 *      array with `cache_control: ephemeral`, matching the direct Anthropic
 *      path. The old shim flattened it to a plain string, losing the cache
 *      discount on every call.
 *   3. **Named errors.** An unsupported combination raises a typed error whose
 *      message names the fix, instead of a 400 from the endpoint.
 *
 * Two deliberate differences from the retired `callOpenAICompat`, both fixes:
 *   - No unconditional `temperature: 0.7`. Models that reject sampling params
 *     (Opus 4.8 and friends) got an HTTP 400 from that.
 *   - The model id is sent verbatim. `'anthropic/claude-sonnet-4-6'` is a valid
 *     OpenRouter slug; stripping the vendor prefix produced an id the endpoint
 *     does not know.
 */
async function callOpenRouterProvider(
  input: AnthropicCallInput & {
    apiKey: string;
    baseUrl: string;
    defaultModel: string;
  },
): Promise<AnthropicCallResult> {
  const startedAt = Date.now();
  const modelId = resolveOpenRouterModel(input.model, input.defaultModel);

  // Lazy, and deliberately so. A top-level static import would make every CLI
  // entry point load the whole provider system merely because this module is
  // reachable from the mcp-tools registry — and would break any harness that
  // exercises a subset of the source tree in isolation.
  let OpenRouterProvider: typeof import('@claude-flow/providers').OpenRouterProvider;
  try {
    ({ OpenRouterProvider } = await import('@claude-flow/providers'));
  } catch (err) {
    return {
      success: false,
      model: modelId,
      error:
        `The OpenRouter provider module could not be loaded: ${err instanceof Error ? err.message : String(err)}. ` +
        'Install @claude-flow/providers, or set ANTHROPIC_API_KEY to use the direct Anthropic path.',
      durationMs: Date.now() - startedAt,
    };
  }

  // A module can resolve successfully and still be the wrong shape. The
  // published @claude-flow/providers@3.0.0 tarball predates this provider: it
  // imports fine and simply has no OpenRouterProvider export. Without this
  // guard the construction below throws a raw TypeError out of
  // callAnthropicMessages, whose contract is to always return an
  // AnthropicCallResult — the throw would skip executeAgentTask's ADR-149
  // fallback and health recording, and the named error above would never fire
  // in the one case it exists for.
  if (typeof OpenRouterProvider !== 'function') {
    return {
      success: false,
      model: modelId,
      error:
        'The installed @claude-flow/providers does not export OpenRouterProvider, so it is too old. ' +
        'Upgrade @claude-flow/providers to a version that ships the OpenRouter provider, ' +
        'or set ANTHROPIC_API_KEY to use the direct Anthropic path.',
      durationMs: Date.now() - startedAt,
    };
  }

  const request: LLMRequest = {
    model: modelId,
    maxTokens: input.maxTokens || 1024,
    messages: [
      ...(input.systemPrompt ? [{ role: 'system' as const, content: input.systemPrompt }] : []),
      { role: 'user' as const, content: input.prompt },
    ],
    ...(typeof input.temperature === 'number' ? { temperature: input.temperature } : {}),
    ...(Array.isArray(input.tools) && input.tools.length > 0
      ? { tools: input.tools as LLMRequest['tools'] }
      : {}),
  };

  try {
    const provider = new OpenRouterProvider({
      // Silenced deliberately: the default consoleLogger prints an INFO line on
      // every initialize(), which would corrupt JSON output for `--format json`
      // consumers. Diagnostics here come from the returned result, not stdout.
      logger: { info: () => {}, warn: () => {}, error: () => {}, debug: () => {} },
      config: {
        provider: 'openrouter',
        apiKey: input.apiKey,
        apiUrl: input.baseUrl,
        model: modelId,
        maxTokens: input.maxTokens || 1024,
        // Honour the caller's deadline. Without this the provider falls back to
        // its own 60s default and a stalled response body is never aborted — the
        // caller's timeout silently stops working.
        ...(typeof input.timeoutMs === 'number' ? { timeout: input.timeoutMs } : {}),
        // The caller's explicit temperature only. Never a default: a model that
        // declares temperature unsupported must not receive one.
        ...(typeof input.temperature === 'number' ? { temperature: input.temperature } : {}),
        providerOptions: {
          cwd: getProjectCwd(),
          // One call per execution: skip the liveness probe that would otherwise
          // add a wasted /v1/models round trip before the completion.
          skipInitialHealthCheck: true,
        },
      },
    });

    await provider.initialize();
    const res: LLMResponse = await provider.complete(request);
    return {
      success: true,
      model: res.model as string,
      messageId: res.id,
      stopReason: res.finishReason ?? 'end_turn',
      output: res.content,
      usage: {
        inputTokens: res.usage?.promptTokens ?? 0,
        outputTokens: res.usage?.completionTokens ?? 0,
        totalTokens: res.usage?.totalTokens ?? 0,
      },
      durationMs: Date.now() - startedAt,
    };
  } catch (err) {
    // Surface the named error's own message. Its text names the fix ("declare
    // it under agents.modelCapabilities", "remove the tools", ...), which a
    // generic wrapper would throw away.
    //
    // The HTTP status MUST be included. The ADR-149 fallback router decides
    // whether to try another model by testing this string for a transient
    // transport failure (429/5xx/timeout). Reporting only the API's own
    // message text silently disables retries for a provider that returns 503
    // with a bare "unavailable" body.
    const status = (err as { statusCode?: number } | null)?.statusCode;
    // Cap the text, as the retired shim did. An upstream body can be a whole
    // HTML page; it must not land unbounded in a result object.
    const raw = err instanceof Error ? err.message : String(err);
    const message = raw.length > 400 ? `${raw.slice(0, 400)}…` : raw;
    const wrapped = (err as { originalError?: unknown } | null)?.originalError;
    const cause = wrapped instanceof Error && wrapped.message && !message.includes(wrapped.message)
      ? ` (cause: ${wrapped.message.slice(0, 200)})`
      : '';
    // Prefer the provider's own classification. RufloProviderError carries
    // `retryable` explicitly (false for auth/capability/permanent faults), and
    // LLMProviderError carries `statusCode`. Aborts arrive with no status but
    // are retryable, which is precisely what the text-regex could not express.
    const isAbort = err instanceof Error && (err.name === 'AbortError' || err.name === 'TimeoutError');
    const retryable = typeof (err as { retryable?: boolean } | null)?.retryable === 'boolean'
      ? (err as { retryable: boolean }).retryable
      : isAbort || typeof status === 'number'
        ? isAbort || status === 408 || status === 429 || (typeof status === 'number' && status >= 500)
        : false;
    return {
      success: false,
      model: modelId,
      error: typeof status === 'number'
        ? `openrouter API error ${status}: ${message}${cause}`
        : `${message}${cause}`,
      ...(typeof status === 'number' ? { statusCode: status } : {}),
      retryable,
      durationMs: Date.now() - startedAt,
    };
  }
}

/**
 * Resolve a caller model id to an OpenRouter slug.
 *
 * Tier aliases map to concrete slugs. Everything else — including
 * `'anthropic/…'` ids — passes through unchanged: those ARE valid OpenRouter
 * model ids, and stripping the vendor prefix (what the old code did) produces
 * an id the endpoint rejects.
 */
function resolveOpenRouterModel(input: string | undefined, fallback: string): string {
  if (!input) return fallback;
  if (input === 'haiku') return 'anthropic/claude-haiku-4-5';
  if (input === 'sonnet' || input === 'inherit') return 'anthropic/claude-sonnet-4-6';
  if (input === 'opus') return 'anthropic/claude-opus-4-8';
  return input;
}

/**
 * Resolve the model name to send to an Ollama-compatible endpoint.
 *
 * Self-hosted setups name their default via OLLAMA_DEFAULT_MODEL or
 * `providers configure -p ollama -m <tag>` (`defaultModel`, resolved by the
 * caller). When no default is configured AND the endpoint is not actually
 * Ollama's real cloud service, this must NOT silently rewrite a tier-routed
 * `claude-*` id (or a bare tier alias) to a `-cloud`-suffixed model name: a
 * signed-in local Ollama daemon forwards any `-cloud`-suffixed model to
 * Ollama's actual cloud service, so a silent rewrite would leak a prompt
 * that was supposed to stay local. Fail loudly instead and tell the caller
 * how to fix it — a loud, local failure beats silent cloud egress.
 */
function resolveOllamaModel(input: string | undefined, defaultModel?: string, isCloudEndpoint = false): string {
  const cloudDefault = 'gpt-oss:120b-cloud';
  const resolveDefault = (): string => {
    if (defaultModel) return defaultModel;
    if (isCloudEndpoint) return cloudDefault;
    throw new Error(
      'No Ollama model configured for this self-hosted/local endpoint — refusing to fall back to '
      + `a cloud-suffixed default ("${cloudDefault}") a local Ollama daemon would forward to Ollama `
      + 'Cloud. Set OLLAMA_DEFAULT_MODEL, or run `providers configure -p ollama -m <tag>`, to pin a '
      + 'real local model.',
    );
  };
  if (!input) return resolveDefault();
  // Logical tier alias → default
  if (input === 'haiku' || input === 'sonnet' || input === 'opus' || input === 'inherit') {
    return resolveDefault();
  }
  // Explicit provider prefix
  if (input.startsWith('ollama:')) return input.slice('ollama:'.length);
  // Anthropic model ids (what executeAgentTask sends for a tier-routed agent
  // via MODEL_MAP) never exist on an Ollama endpoint → default.
  if (/^claude-/i.test(input)) return resolveDefault();
  // Bare name (e.g. 'qwen3-coder:30b', 'llama3:70b-cloud') passes through
  return input;
}

/**
 * Resolve a model identifier to an Anthropic model ID. Accepts:
 * - logical names: 'haiku', 'sonnet', 'opus', 'inherit'
 * - prefixed: 'anthropic:claude-sonnet-4-6'
 * - direct: 'claude-sonnet-4-6'
 */
export function resolveAnthropicModel(input: string | undefined): string {
  if (!input) return DEFAULT_ANTHROPIC_MODEL;
  if (input in MODEL_MAP) return MODEL_MAP[input];
  if (input.startsWith('anthropic:')) return input.slice('anthropic:'.length);
  return input;
}

export interface AgentExecuteInput {
  agentId: string;
  prompt: string;
  systemPrompt?: string;
  maxTokens?: number;
  temperature?: number;
  timeoutMs?: number;
}

export interface AgentExecuteResult {
  success: boolean;
  agentId: string;
  model?: string;
  messageId?: string;
  stopReason?: string;
  output?: string;
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number };
  durationMs?: number;
  error?: string;
  remediation?: string;
  /**
   * ADR-149 iter 7 — present when the request was retried after a 429/5xx
   * via `nextCostOptimalAlternative`. Each entry records a model that
   * was tried and failed, in attempt order. The final `model` field is
   * the one that produced the surfaced result (success or final error).
   */
  fallbackHistory?: Array<{ modelId: string; error: string }>;
}

// Tracks overlapping calls in this process; persisted state remains authoritative.
const activeExecutions = new Map<string, number>();

export async function executeAgentTask(input: AgentExecuteInput): Promise<AgentExecuteResult> {
  const store = loadAgentStore();
  const agent = store.agents[input.agentId];
  if (!agent) return { success: false, agentId: input.agentId, error: 'Agent not found' };
  if (agent.status === 'terminated') return { success: false, agentId: input.agentId, error: 'Agent has been terminated' };

  // ADR-149 iter 13 — first-call dispatch prefers `agent.modelId` (the
  // cost-optimal pick from the neural backend) over `MODEL_MAP[agent.model]`
  // when present. Before this fix the first attempt always used the tier
  // mapping; only the iter-7 fallback chain used modelId on retry, which
  // meant the cost-optimal pick was wasted unless the first call failed
  // with a 429/5xx.
  //
  // Rules:
  //   - agent.modelId is set AND it's a non-Anthropic slug (e.g.
  //     'inclusionai/ling-2.6-flash', 'openai/gpt-4.1') → dispatch
  //     directly via that id. callAnthropicMessages forwards non-Anthropic
  //     ids through OpenRouter (#2042), so this gets us the cost-optimal
  //     model on attempt 1.
  //   - agent.modelId starts with 'anthropic/' → strip the prefix, use the
  //     bare Anthropic id (e.g. 'anthropic/claude-haiku-4.5' → 'claude-haiku-4.5').
  //   - agent.modelId is unset (no neural backend fired) → fall back to the
  //     legacy tier-mapped MODEL_MAP path. Existing behaviour preserved.
  let firstCallModel: string;
  if (agent.modelId) {
    firstCallModel = agent.modelId.startsWith('anthropic/')
      ? agent.modelId.slice('anthropic/'.length)
      : agent.modelId;
  } else {
    firstCallModel = resolveAnthropicModel(agent.model || 'sonnet');
  }
  // Kept for legacy error-path remediation message + final-result `model` field
  // (returned when the request fully fails with no successful retry).
  const anthropicModel = firstCallModel;
  // The agent registry persists per-agent instructions, including those set
  // through agent_update. Workflow steps do not supply a systemPrompt, so
  // omitting this fallback silently discards the configured agent context.
  const configuredInstructions = typeof agent.config?.instructions === 'string' && agent.config.instructions.trim()
    ? agent.config.instructions
    : undefined;
  const systemPrompt = input.systemPrompt ||
    configuredInstructions ||
    `You are a ${agent.agentType} agent operating as part of a Ruflo swarm. ` +
    `Agent ID: ${input.agentId}. Domain: ${agent.domain ?? 'general'}. ` +
    `Respond directly and stay focused on the task. If you need information you don't have, state that explicitly.`;

  const initialModelId = agent.modelId;
  const initialModel = agent.model;
  agent.status = 'busy';
  agent.taskCount = (agent.taskCount || 0) + 1;
  saveAgentStore(store);

  const executionKey = JSON.stringify([getAgentPath(), input.agentId, agent.createdAt]);
  activeExecutions.set(executionKey, (activeExecutions.get(executionKey) || 0) + 1);
  try {
  const startedAt = Date.now();

  // #2042 — delegate to callAnthropicMessages so the v3 provider router
  // (Anthropic / Ollama / OpenRouter) governs which backend is hit.
  // #2962 — forward the agent's own explicit provider (set at spawn time
  // from --provider / persisted config) so it outranks env-var inference.
  // Only the first-attempt call; the ADR-149 fallback-retry call below is
  // driven by the cost-optimal neural router picking model-id alternatives,
  // an orthogonal mechanism left as-is to avoid unintended interaction.
  let result = await callAnthropicMessages({
    model: anthropicModel,
    prompt: input.prompt,
    systemPrompt,
    maxTokens: input.maxTokens,
    temperature: input.temperature,
    timeoutMs: input.timeoutMs,
    provider: agent.provider,
  });

  // ADR-149 iter 7 — fallback chain on retryable failures (429, 5xx,
  // timeout). When the cost-optimal neural backend picked a specific
  // model id and that model fails for a transient reason, fall back to
  // the next-cheapest candidate that clears the quality bar. Budget is
  // bounded by CLAUDE_FLOW_ROUTER_FALLBACK_MAX_RETRIES (default 1) so
  // upstream outages don't cause retry storms.
  const configuredFallbackBudget = parseInt(process.env.CLAUDE_FLOW_ROUTER_FALLBACK_MAX_RETRIES ?? '1', 10);
  const fallbackBudget = Number.isNaN(configuredFallbackBudget) ? 1 : Math.max(0, configuredFallbackBudget);
  const fallbackHistory: Array<{ modelId: string; error: string }> = [];
  // Keep feedback for each completed attempt before switching model identity.
  const failedAttempts: Array<{ model: ClaudeModel; modelId: string }> = [];
  if (!result.success && agent.modelId && fallbackBudget > 0) {
    const isRetryable = isRetryableFailure(result);
    if (isRetryable) {
      try {
        const { nextCostOptimalAlternative } = await import('../ruvector/neural-router.js');
        // ADR-149 iter 9 — delegate to the shared task-embedder LRU. The
        // pipeline + cache are shared with agent-tools.ts, so the embedding
        // for this prompt is almost always already cached from the initial
        // routing decision (no extra inference cost in steady state).
        const { embedTaskWithCache } = await import('../ruvector/task-embedder.js');
        const embedding = await embedTaskWithCache(input.prompt);
        if (embedding) {
          const excludeIds: string[] = [agent.modelId];
          for (let attempt = 0; attempt < fallbackBudget && !result.success; attempt++) {
            fallbackHistory.push({ modelId: excludeIds[excludeIds.length - 1], error: result.error ?? '' });
            const alt = await nextCostOptimalAlternative(embedding, excludeIds);
            if (!alt || !alt.modelId) break;
            excludeIds.push(alt.modelId);
            const altResult = await callAnthropicMessages({
              model: alt.modelId,
              prompt: input.prompt,
              systemPrompt,
              maxTokens: input.maxTokens,
              temperature: input.temperature,
              timeoutMs: input.timeoutMs,
              // N-3 — keep the agent's provider pin. Without this the retry
              // re-resolves from env and can silently land on a different
              // provider than the one the agent was pinned to.
              provider: agent.provider,
            });
            // Record the model that ACTUALLY answered (or errored). On success,
            // update agent.modelId so downstream observers see the retry winner.
            failedAttempts.push({ model: agent.model ?? 'sonnet', modelId: agent.modelId! });
            agent.modelId = alt.modelId;
            agent.model = alt.model;
            result = altResult;
          }
        }
      } catch {
        // Fallback chain is best-effort — preserve the original error result.
      }
    }
  }


  // ADR-149 — close the bandit feedback loop. `recordModelOutcome` updates
  // the Beta(α,β) prior for the agent's tier so the Thompson sampler learns
  // from production traffic instead of staying frozen at install-day priors.
  // This is best-effort: any error here must NOT break the agent execution.
  // For now, "success" = the model returned a response without an API error.
  // A finer-grained signal (user-accepted output / regression-detected) is a
  // follow-up; this commit closes the bandit's basic learning loop.
  try {
    const { recordModelOutcome, recordModelOutcomeByModelId } = await import('../ruvector/model-router.js');
    // Bandit priors are keyed on the 3 canonical tiers (haiku/sonnet/opus/inherit);
    // collapse opus-4.7 → opus before recording so the bandit's per-tier Beta
    // updates correctly.
    const tier: 'haiku' | 'sonnet' | 'opus' | 'inherit' =
      agent.model === 'opus-4.7' ? 'opus' :
      (agent.model as 'haiku' | 'sonnet' | 'opus' | 'inherit' | undefined) ?? 'sonnet';
    const outcome: 'success' | 'failure' = result.success ? 'success' : 'failure';
    for (const attempt of failedAttempts) {
      recordModelOutcome(input.prompt, attempt.model === 'opus-4.7' ? 'opus' : attempt.model, 'failure');
      recordModelOutcomeByModelId(input.prompt, attempt.modelId, 'failure');
    }
    recordModelOutcome(input.prompt, tier, outcome);
    // ADR-149 — also write to the shadow per-modelId priors when the cost-
    // optimal neural backend picked a concrete model id. Selection logic
    // still uses tier priors, but the per-modelId data accumulates so a
    // future refactor can switch the selector over.
    if (agent.modelId) {
      recordModelOutcomeByModelId(input.prompt, agent.modelId, outcome);
    }
    // ADR-149 iter 17 — close the production-data side of the feedback loop.
    // The trajectory recorder (CLAUDE_FLOW_ROUTER_TRAJECTORY=1) writes
    // per-decision rows to .swarm/model-router-trajectories.jsonl from the
    // model-router itself. Now we ALSO write outcome rows here so future
    // training (scripts/train-from-trajectories.mjs, follow-up) can pair
    // decision+outcome by task_hash and produce DRACO-shaped retraining
    // rows from real production traffic. quality = 1.0 on success, 0.0 on
    // failure (coarse signal; finer-grained quality from user ratings or
    // regression-detection is a separate hook).
    if (process.env.CLAUDE_FLOW_ROUTER_TRAJECTORY === '1') {
      try {
        const { recordTrajectoryOutcome } = await import('../ruvector/router-trajectory.js');
        const scores: Record<string, number> | undefined = agent.modelId
          ? { [agent.modelId]: outcome === 'success' ? 1.0 : 0.0 }
          : undefined;
        // iter 31 — pass token usage + modelId so the recorder can compute
        // USD spend at write time. Consumers (`router decisions`, cost-
        // savings reports) get cost without their own price table.
        recordTrajectoryOutcome({
          task: input.prompt,
          quality: outcome === 'success' ? 1.0 : 0.0,
          scores,
          source: 'agent-execute',
          tokens: result.usage ? { input: result.usage.inputTokens, output: result.usage.outputTokens } : undefined,
          modelId: agent.modelId,
        });
      } catch { /* never break execution */ }
    }
    // ADR-150 weight-eft capture seam — the ONLY place a run's full outcome is
    // known here: prompt (issue), assistant transcript, model, tier, and a
    // resolved PROXY. Gated behind CLAUDE_FLOW_RUN_TRANSCRIPTS=1 (off by
    // default; PII/retention surface, mirrors the router trajectory recorder).
    // HONESTY: `resolved` here is the WEAKEST proxy — 'api-success' means only
    // that the model returned without an API error, NOT that the output is
    // correct (ruflo has no SWE-bench gold oracle). model_patch is '' because
    // this single-shot execute path produces no unified diff. Both facts are
    // stamped so no downstream weight-eft export mistakes this for gold data.
    if (process.env.CLAUDE_FLOW_RUN_TRANSCRIPTS === '1') {
      try {
        const { recordRunTranscript, tierForModel } = await import('../ruvector/run-transcript-recorder.js');
        const modelId = agent.modelId ?? String(agent.model ?? 'unknown');
        const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [];
        if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
        messages.push({ role: 'user', content: input.prompt });
        if (result.success && typeof result.output === 'string') {
          messages.push({ role: 'assistant', content: result.output });
        }
        recordRunTranscript({
          task: input.prompt,
          model: modelId,
          tier: tierForModel(modelId),
          resolved: outcome === 'success',
          resolvedSource: 'api-success',
          messages,
          source: 'agent-execute',
          tokens: result.usage ? { input: result.usage.inputTokens, output: result.usage.outputTokens } : undefined,
        });
      } catch { /* never break execution */ }
    }
  } catch {
    // Silent — bandit feedback must never block routing.
  }

  const remaining = (activeExecutions.get(executionKey) || 1) - 1;
  const persistCompletion = (out?: AgentExecuteResult): void => {
    // Awaited provider/feedback work may have overlapped registration, updates,
    // termination or deletion. Never replace that newer registry snapshot.
    const latest = loadAgentStore();
    const current = latest.agents[input.agentId];
    if (!current || current.createdAt !== agent.createdAt) return;
    if (current.status === 'busy') current.status = remaining > 0 ? 'busy' : 'idle';
    if (agent.modelId !== initialModelId && current.modelId === initialModelId) {
      // A fallback may also change its tier. Carry that change with the model
      // ID, but preserve any selection the user edited during the request.
      if (agent.model !== initialModel && current.model === initialModel) current.model = agent.model;
      current.modelId = agent.modelId;
    }
    if (out) current.lastResult = out as unknown as Record<string, unknown>;
    const previousMeasuredHealth = summarizeAgentExecutions(current.executionHistory).health;
    const mayUpdateStoredHealth = previousMeasuredHealth === null
      ? current.health === agent.health
      : current.health === previousMeasuredHealth;
    current.executionHistory = appendAgentExecution(current.executionHistory, {
      at: new Date().toISOString(),
      success: result.success,
      durationMs: result.durationMs ?? Date.now() - startedAt,
    });
    // An explicit agent_update during the provider call still wins. The
    // measured value remains available from executionHistory in agent_health.
    if (mayUpdateStoredHealth) {
      current.health = summarizeAgentExecutions(current.executionHistory).health ?? current.health;
    }
    saveAgentStore(latest);
  };

  if (result.success) {
    const out: AgentExecuteResult = {
      success: true,
      agentId: input.agentId,
      messageId: result.messageId,
      model: result.model,
      stopReason: result.stopReason,
      output: result.output,
      usage: result.usage,
      durationMs: result.durationMs ?? Date.now() - startedAt,
      ...(fallbackHistory.length > 0 ? { fallbackHistory } : {}),
    };
    persistCompletion(out);
    return out;
  }

  persistCompletion();
  // No-provider-configured error → surface the same actionable message
  // the router built, with a #2042-aware remediation pointer.
  const noProvider = (result.error || '').includes('No LLM provider configured');
  return {
    success: false,
    agentId: input.agentId,
    model: anthropicModel,
    error: result.error || 'agent_execute failed',
    durationMs: result.durationMs ?? Date.now() - startedAt,
    ...(fallbackHistory.length > 0 ? { fallbackHistory } : {}),
    ...(noProvider && {
      remediation:
        'Set one of ANTHROPIC_API_KEY, OPENROUTER_API_KEY (+ optional OPENROUTER_BASE_URL), or OLLAMA_API_KEY. ' +
        'Or set RUFLO_PROVIDER=openrouter|ollama to force a specific provider.',
    }),
  };
  } finally {
    const pending = (activeExecutions.get(executionKey) || 1) - 1;
    if (pending > 0) activeExecutions.set(executionKey, pending);
    else activeExecutions.delete(executionKey);
  }
}
