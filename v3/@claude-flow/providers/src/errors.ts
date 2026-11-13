/**
 * V3 Provider Error Hierarchy
 *
 * Named, catchable errors for capability mismatches, provider/credential
 * faults, and agent-loop limits.
 *
 * Design rule: unsupported combinations fail loudly and pre-flight. Never
 * silently drop a parameter or swap a model.
 *
 * @module @claude-flow/providers/errors
 */

import { LLMProviderError } from './types.js';
import type { LLMProvider } from './types.js';

/**
 * Base for every provider-side error. Carries machine-readable fields so
 * callers can branch on `code` without string-matching a message.
 *
 * NOTE: this extends LLMProviderError on purpose. BaseProvider.transformError
 * passes LLMProviderError instances through untouched and rewraps everything
 * else as UNKNOWN — extending it is what keeps the specific error code alive
 * all the way out to the caller.
 */
export class RufloProviderError extends LLMProviderError {
  readonly modelId?: string;
  readonly capability?: string;

  constructor(
    message: string,
    opts: {
      code: string;
      provider: LLMProvider;
      modelId?: string;
      capability?: string;
      retryable?: boolean;
    }
  ) {
    super(message, opts.code, opts.provider, undefined, opts.retryable ?? false);
    this.name = new.target.name;
    this.modelId = opts.modelId;
    this.capability = opts.capability;
    // Keep instanceof working across transpilation targets.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** The requested feature is not available on this model/provider. */
export class UnsupportedCapabilityError extends RufloProviderError {
  constructor(opts: {
    code: string;
    provider: LLMProvider;
    modelId?: string;
    capability: string;
    message: string;
  }) {
    super(opts.message, {
      code: opts.code,
      provider: opts.provider,
      modelId: opts.modelId,
      capability: opts.capability,
      retryable: false,
    });
  }
}

/** A sampling parameter was supplied to a model that rejects it. */
export class UnsupportedParameterError extends RufloProviderError {
  constructor(opts: {
    provider: LLMProvider;
    modelId?: string;
    parameter: string;
    message?: string;
  }) {
    super(
      opts.message ??
        `Model "${opts.modelId ?? 'unknown'}" rejects the sampling parameter ` +
          `"${opts.parameter}". Remove it from the request, or pin a model that accepts it.`,
      {
        code: 'ERR_PARAM_SAMPLING_REJECTED',
        provider: opts.provider,
        modelId: opts.modelId,
        capability: 'samplingParams',
        retryable: false,
      }
    );
  }
}

/** A cloud service is being asked to run on a provider that cannot serve it. */
export class ServiceProviderMismatchError extends RufloProviderError {
  constructor(opts: {
    code?: string;
    provider: LLMProvider;
    service: string;
    message?: string;
  }) {
    super(
      opts.message ??
        `${opts.service} is an Anthropic-hosted cloud service and cannot run through ` +
          `provider "${opts.provider}". Set ANTHROPIC_API_KEY and use the direct Anthropic ` +
          `provider, or use a different tool.`,
      {
        code: opts.code ?? 'ERR_MANAGED_AGENTS_REQUIRES_ANTHROPIC_DIRECT',
        provider: opts.provider,
        capability: 'managedAgentsService',
        retryable: false,
      }
    );
  }
}

/** No usable credential was found for the selected provider. */
export class ProviderAuthenticationError extends RufloProviderError {
  constructor(opts: { provider: LLMProvider; message?: string }) {
    super(
      opts.message ??
        `No credential configured for provider "${opts.provider}". Set the matching API key environment variable.`,
      {
        code: 'ERR_PROVIDER_NO_CREDENTIALS',
        provider: opts.provider,
        capability: 'credentials',
        retryable: false,
      }
    );
  }
}

/** The model could not be resolved to a known capability profile. */
export class UnknownModelCapabilityError extends RufloProviderError {
  constructor(opts: { provider: LLMProvider; modelId: string; message?: string }) {
    super(
      opts.message ??
        `No capability profile is known for model "${opts.modelId}". Declare it under ` +
          `agents.modelCapabilities in claude-flow.config.json, or run ` +
          `"ruflo providers models --refresh" to fetch the OpenRouter catalog.`,
      {
        code: 'ERR_MODEL_CAPABILITY_UNVERIFIED',
        provider: opts.provider,
        modelId: opts.modelId,
        capability: 'capabilityProfile',
        retryable: false,
      }
    );
  }
}

/** An agent loop hit its tool-call iteration ceiling without finishing. */
export class AgentLoopMaxIterationsError extends RufloProviderError {
  constructor(opts: { provider: LLMProvider; modelId?: string; iterations: number }) {
    super(
      `Agent loop reached ${opts.iterations} tool iterations without a terminal response. ` +
        `Raise maxToolIterations, or check that the tool executor returns usable results.`,
      {
        code: 'ERR_AGENT_LOOP_MAX_ITERATIONS',
        provider: opts.provider,
        modelId: opts.modelId,
        capability: 'tools',
        retryable: false,
      }
    );
  }
}

/** Stable capability-error codes, exported so callers avoid magic strings. */
export const CAPABILITY_CODES = {
  TOOLS: 'ERR_CAPABILITY_TOOLS_UNSUPPORTED',
  STRUCTURED_OUTPUT: 'ERR_CAPABILITY_STRUCTURED_OUTPUT_UNSUPPORTED',
  PROMPT_CACHE: 'ERR_CAPABILITY_PROMPT_CACHE_UNSUPPORTED',
  REASONING: 'ERR_CAPABILITY_REASONING_UNSUPPORTED',
  VISION: 'ERR_CAPABILITY_VISION_UNSUPPORTED',
  AUDIO: 'ERR_CAPABILITY_AUDIO_UNSUPPORTED',
} as const;
