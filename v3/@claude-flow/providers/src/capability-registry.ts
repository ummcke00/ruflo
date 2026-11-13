/**
 * V3 Model Capability Registry
 *
 * Declared capabilities, never inferred from a model name.
 *
 * Resolution order (first hit wins):
 *   1. User config overrides  — `agents.modelCapabilities[modelId]`
 *   2. OpenRouter catalog cache — `.claude-flow/cache/openrouter-models.json`
 *   3. Static curated manifest  — STATIC_CAPABILITY_MANIFEST below
 *   4. Conservative unverified default — source: 'unverified_default'
 *
 * @module @claude-flow/providers/capability-registry
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import type { LLMProvider } from './types.js';

/** How a model exposes schema-constrained output, if at all. */
export type StructuredOutputMode =
  | 'json_schema'
  | 'json_object'
  | 'tool_strict_fallback'
  | 'unsupported';

/** How prompt caching is expressed on the wire, if at all. */
export type PromptCachingMode =
  | 'anthropic_ephemeral'
  | 'openrouter_multipart_ephemeral'
  | 'implicit_prefix'
  | 'unsupported';

/** How reasoning effort is expressed on the wire, if at all. */
export type ReasoningControlMode =
  | 'anthropic_thinking'
  | 'openrouter_reasoning_effort'
  | 'openrouter_reasoning_max_tokens'
  | 'unsupported';

export type SupportFlag = 'supported' | 'unsupported';

/** Where a resolved profile came from — surfaced for debugging and errors. */
export type CapabilitySource =
  | 'config_override'
  | 'openrouter_catalog'
  | 'static_manifest'
  | 'unverified_default';

/**
 * The declared capability record for one (provider, modelId) pair.
 */
export interface ModelCapabilityProfile {
  modelId: string;
  provider: LLMProvider;

  tools: SupportFlag;
  parallelToolCalls: boolean;
  structuredOutput: StructuredOutputMode;
  promptCaching: PromptCachingMode;
  reasoningControl: ReasoningControlMode;

  /** Which sampling parameters this model accepts. */
  samplingParams: {
    temperature: boolean;
    topP: boolean;
    topK: boolean;
  };

  vision: SupportFlag;
  audio: SupportFlag;

  /** Only ever 'supported' for the direct Anthropic provider. */
  managedAgentsService: SupportFlag;

  maxContextLength?: number;
  maxOutputTokens?: number;

  source: CapabilitySource;
}

/** Conservative profile for anything we have not verified. */
export function unverifiedDefault(
  provider: LLMProvider,
  modelId: string
): ModelCapabilityProfile {
  return {
    modelId,
    provider,
    tools: 'unsupported',
    parallelToolCalls: false,
    structuredOutput: 'unsupported',
    promptCaching: 'unsupported',
    reasoningControl: 'unsupported',
    samplingParams: { temperature: true, topP: true, topK: false },
    vision: 'unsupported',
    audio: 'unsupported',
    managedAgentsService: 'unsupported',
    source: 'unverified_default',
  };
}

/** Convenience constructor so the manifest below stays readable. */
function profile(
  modelId: string,
  overrides: Partial<Omit<ModelCapabilityProfile, 'modelId' | 'source'>> = {}
): ModelCapabilityProfile {
  return {
    ...unverifiedDefault('openrouter', modelId),
    tools: 'supported',
    parallelToolCalls: true,
    structuredOutput: 'json_schema',
    promptCaching: 'implicit_prefix',
    reasoningControl: 'unsupported',
    vision: 'supported',
    audio: 'unsupported',
    ...overrides,
    modelId,
    source: 'static_manifest',
  };
}

/**
 * Curated seed table for models we know. Kept deliberately small and exact —
 * a wrong entry is worse than a missing one, because a missing entry fails
 * loudly while a wrong entry fails silently.
 */
export const STATIC_CAPABILITY_MANIFEST: Record<string, ModelCapabilityProfile> = {
  // ---- Anthropic via OpenRouter -------------------------------------------------
  'anthropic/claude-sonnet-4-6': profile('anthropic/claude-sonnet-4-6', {
    promptCaching: 'openrouter_multipart_ephemeral',
    reasoningControl: 'openrouter_reasoning_max_tokens',
  }),
  'anthropic/claude-opus-4-7': profile('anthropic/claude-opus-4-7', {
    promptCaching: 'openrouter_multipart_ephemeral',
    reasoningControl: 'openrouter_reasoning_max_tokens',
    samplingParams: { temperature: false, topP: false, topK: false },
  }),
  'anthropic/claude-opus-4-8': profile('anthropic/claude-opus-4-8', {
    promptCaching: 'openrouter_multipart_ephemeral',
    reasoningControl: 'openrouter_reasoning_max_tokens',
    samplingParams: { temperature: false, topP: false, topK: false },
  }),
  'anthropic/claude-sonnet-5': profile('anthropic/claude-sonnet-5', {
    promptCaching: 'openrouter_multipart_ephemeral',
    reasoningControl: 'openrouter_reasoning_max_tokens',
    samplingParams: { temperature: false, topP: false, topK: false },
  }),
  'anthropic/claude-fable-5': profile('anthropic/claude-fable-5', {
    promptCaching: 'openrouter_multipart_ephemeral',
    reasoningControl: 'openrouter_reasoning_max_tokens',
    samplingParams: { temperature: false, topP: false, topK: false },
  }),

  // ---- OpenAI via OpenRouter ----------------------------------------------------
  'openai/gpt-4.1': profile('openai/gpt-4.1'),
  'openai/gpt-4o': profile('openai/gpt-4o', { audio: 'supported' }),
  'openai/o3-mini': profile('openai/o3-mini', {
    reasoningControl: 'openrouter_reasoning_effort',
    samplingParams: { temperature: false, topP: false, topK: false },
    vision: 'unsupported',
  }),
  'openai/o4-mini': profile('openai/o4-mini', {
    reasoningControl: 'openrouter_reasoning_effort',
    samplingParams: { temperature: false, topP: false, topK: false },
  }),

  // ---- Google via OpenRouter ----------------------------------------------------
  'google/gemini-2.5-pro': profile('google/gemini-2.5-pro', {
    reasoningControl: 'openrouter_reasoning_max_tokens',
    audio: 'supported',
  }),
  'google/gemini-2.5-flash': profile('google/gemini-2.5-flash', {
    reasoningControl: 'openrouter_reasoning_max_tokens',
    audio: 'supported',
  }),

  // ---- Open-weight via OpenRouter ----------------------------------------------
  'deepseek/deepseek-chat': profile('deepseek/deepseek-chat', { vision: 'unsupported' }),
  'deepseek/deepseek-r1': profile('deepseek/deepseek-r1', {
    reasoningControl: 'openrouter_reasoning_effort',
    vision: 'unsupported',
  }),
  'qwen/qwen3-coder': profile('qwen/qwen3-coder', { vision: 'unsupported' }),
  'meta-llama/llama-3.3-70b-instruct': profile('meta-llama/llama-3.3-70b-instruct', {
    vision: 'unsupported',
    parallelToolCalls: false,
  }),
};

/**
 * Direct-Anthropic profiles. The `provider: 'anthropic'` path is treated
 * separately because it speaks the native Messages API rather than the
 * OpenAI-compatible surface.
 */
function anthropicDirectProfile(modelId: string): ModelCapabilityProfile {
  const rejectsSampling = /^claude-(fable-5|opus-4-8|opus-4-7|sonnet-5)/.test(modelId);
  return {
    ...unverifiedDefault('anthropic', modelId),
    tools: 'supported',
    parallelToolCalls: true,
    structuredOutput: 'json_schema',
    promptCaching: 'anthropic_ephemeral',
    reasoningControl: 'anthropic_thinking',
    samplingParams: {
      temperature: !rejectsSampling,
      topP: !rejectsSampling,
      topK: !rejectsSampling,
    },
    vision: 'supported',
    managedAgentsService: 'supported',
    source: 'static_manifest',
  };
}

/** Shape of one entry in the cached OpenRouter catalog file. */
interface CatalogEntry {
  supportedParameters?: string[];
  contextLength?: number;
  maxOutputTokens?: number;
  inputModalities?: string[];
}

type CatalogCache = Record<string, CatalogEntry>;

/**
 * Read the cached OpenRouter catalog. Returns undefined when absent or
 * unparseable — callers fall through to the static manifest.
 *
 * The fetcher that WRITES this file lands in a later step; this reader is
 * deliberately tolerant of a missing file so the tree works either way.
 */
export function loadOpenRouterModelCatalogCache(cwd: string): CatalogCache | undefined {
  const path = join(cwd, '.claude-flow', 'cache', 'openrouter-models.json');
  try {
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as CatalogCache;
    return parsed && typeof parsed === 'object' ? parsed : undefined;
  } catch {
    return undefined;
  }
}

/** Translate one catalog entry into a capability profile. */
function profileFromCatalog(modelId: string, entry: CatalogEntry): ModelCapabilityProfile {
  const params = new Set(entry.supportedParameters ?? []);
  const modalities = new Set(entry.inputModalities ?? ['text']);

  const hasTools = params.has('tools') || params.has('tool_choice');
  const structured: StructuredOutputMode = params.has('structured_outputs') ||
    params.has('response_format')
    ? 'json_schema'
    : 'unsupported';
  const reasoning: ReasoningControlMode = params.has('reasoning')
    ? 'openrouter_reasoning_max_tokens'
    : 'unsupported';

  return {
    modelId,
    provider: 'openrouter',
    tools: hasTools ? 'supported' : 'unsupported',
    parallelToolCalls: params.has('parallel_tool_calls'),
    structuredOutput: structured,
    // Catalog data does not say where a cache breakpoint belongs; Anthropic
    // models fronted by OpenRouter accept the multipart form, everything else
    // gets implicit prefix caching, which needs no wire change.
    promptCaching: modelId.startsWith('anthropic/')
      ? 'openrouter_multipart_ephemeral'
      : 'implicit_prefix',
    reasoningControl: reasoning,
    samplingParams: {
      temperature: params.has('temperature'),
      topP: params.has('top_p'),
      topK: params.has('top_k'),
    },
    vision: modalities.has('image') ? 'supported' : 'unsupported',
    audio: modalities.has('audio') ? 'supported' : 'unsupported',
    managedAgentsService: 'unsupported',
    maxContextLength: entry.contextLength,
    maxOutputTokens: entry.maxOutputTokens,
    source: 'openrouter_catalog',
  };
}

export interface ResolveCapabilitiesOptions {
  /** Explicit overrides, keyed by model id — normally `agents.modelCapabilities`. */
  overrides?: Record<string, Partial<ModelCapabilityProfile>>;
  /** Working directory used to locate the OpenRouter catalog cache. */
  cwd?: string;
  /** Skip catalog-cache lookup (tests, air-gapped runs). */
  skipCatalog?: boolean;
}

/**
 * Resolve the capability profile for a (provider, modelId) pair.
 *
 * Never throws. Unknown models return an `unverified_default` profile, and
 * the caller is expected to fail loudly via validateRequest() if it then asks
 * for an advanced capability.
 */
export function resolveModelCapabilities(
  provider: LLMProvider,
  modelId: string,
  options: ResolveCapabilitiesOptions = {}
): ModelCapabilityProfile {
  // 1. Explicit overrides always win.
  const override = options.overrides?.[modelId];
  if (override) {
    return {
      ...unverifiedDefault(provider, modelId),
      ...override,
      modelId,
      provider,
      source: 'config_override',
    };
  }

  // 2. Direct Anthropic has its own profile shape.
  if (provider === 'anthropic') {
    return anthropicDirectProfile(modelId);
  }

  // 3. Cached OpenRouter catalog.
  if (provider === 'openrouter' && !options.skipCatalog && options.cwd) {
    const cache = loadOpenRouterModelCatalogCache(options.cwd);
    const entry = cache?.[modelId];
    if (entry) return profileFromCatalog(modelId, entry);
  }

  // 4. Static manifest.
  const seeded = STATIC_CAPABILITY_MANIFEST[modelId];
  if (seeded) return { ...seeded, provider };

  // 5. Nothing known — conservative default.
  return unverifiedDefault(provider, modelId);
}

/** True when the profile carries real evidence rather than a fallback. */
export function isVerifiedProfile(p: ModelCapabilityProfile): boolean {
  return p.source !== 'unverified_default';
}
