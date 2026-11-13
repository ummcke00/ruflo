/**
 * V3 OpenRouter Provider
 *
 * First-class OpenRouter support: any model OpenRouter fronts, driven through
 * one OpenAI-compatible wire surface, with every optional feature gated on a
 * *declared* capability profile rather than guessed from the model name.
 *
 * What this provider does that the old OpenAI-compat shim did not:
 *   - sends `tools` / `tool_choice` and parses `tool_calls` back
 *   - sends `response_format` honouring the declared mode (json_schema vs json_object)
 *   - sends multipart `cache_control` for Anthropic models, plain string for
 *     models that use implicit prefix caching
 *   - forwards `reasoning` in the shape the model declares (effort vs max_tokens)
 *   - omits sampling parameters a model rejects, and throws if the caller
 *     explicitly asked for one anyway
 *   - sets `provider: { require_parameters: true }` only when the request
 *     carries parameters not every upstream serves
 *   - validates vision/audio pre-flight instead of sending unsupported parts
 *
 * @module @claude-flow/providers/openrouter-provider
 */

import { BaseProvider, BaseProviderOptions } from './base-provider.js';
import {
  LLMProvider,
  LLMModel,
  LLMRequest,
  LLMResponse,
  LLMStreamEvent,
  ModelInfo,
  ProviderCapabilities,
  HealthCheckResult,
  LLMToolCall,
  AuthenticationError,
  LLMProviderError,
  ModelNotFoundError,
  RateLimitError,
} from './types.js';
import {
  ModelCapabilityProfile,
  resolveModelCapabilities,
} from './capability-registry.js';
import {
  UnsupportedCapabilityError,
  UnsupportedParameterError,
  UnknownModelCapabilityError,
  CAPABILITY_CODES,
} from './errors.js';

const DEFAULT_BASE_URL = 'https://openrouter.ai/api';
const DEFAULT_MODEL = 'anthropic/claude-sonnet-4-6';

/** Anthropic's hard limit on cache breakpoints per request. */
const MAX_CACHE_BREAKPOINTS = 4;

/** Logical tier aliases resolve to concrete OpenRouter slugs. */
const TIER_ALIASES: Record<string, string> = {
  haiku: 'anthropic/claude-haiku-4-5',
  sonnet: 'anthropic/claude-sonnet-4-6',
  opus: 'anthropic/claude-opus-4-8',
  inherit: DEFAULT_MODEL,
};

/** A canonical tool definition as the caller supplies it. */
export interface OpenRouterToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
    strict?: boolean;
  };
}

/** A schema constraint for structured output. */
export interface OpenRouterResponseSchema {
  name: string;
  schema: Record<string, unknown>;
  strict?: boolean;
}

/**
 * Optional extras this provider accepts beyond the shared LLMRequest.
 *
 * Deliberately no `strict` escape hatch: unsupported combinations always fail
 * loudly. See bundle/decisions.md — a flag that permits silent degradation is
 * how the original bug class returned.
 */
export interface OpenRouterRequestExtras {
  responseSchema?: OpenRouterResponseSchema;
  reasoning?: { effort?: 'low' | 'medium' | 'high'; maxTokens?: number };
}

interface OpenRouterMessage {
  role: string;
  content: string | Array<Record<string, unknown>>;
  name?: string;
  tool_call_id?: string;
  tool_calls?: LLMToolCall[];
}

interface OpenRouterChatRequest {
  model: string;
  messages: OpenRouterMessage[];
  stream: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  tools?: OpenRouterToolDefinition[];
  tool_choice?: unknown;
  response_format?: Record<string, unknown>;
  reasoning?: Record<string, unknown>;
  provider?: Record<string, unknown>;
}

interface OpenRouterChoiceMessage {
  role?: string;
  content?: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
}

interface OpenRouterChatResponse {
  id?: string;
  model?: string;
  choices?: Array<{
    message?: OpenRouterChoiceMessage;
    delta?: OpenRouterChoiceMessage;
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number };
  };
  error?: { message?: string; code?: number };
}

export class OpenRouterProvider extends BaseProvider {
  readonly name: LLMProvider = 'openrouter';

  readonly capabilities: ProviderCapabilities = {
    // OpenRouter fronts hundreds of models; the authoritative list is fetched
    // at runtime in a later step. This seed keeps validateModel() useful offline
    // and covers every tier alias so aliases resolve to verified profiles.
    supportedModels: [
      'anthropic/claude-haiku-4-5',
      'anthropic/claude-sonnet-4-6',
      'anthropic/claude-opus-4-7',
      'anthropic/claude-opus-4-8',
      'anthropic/claude-sonnet-5',
      'openai/gpt-4.1',
      'openai/gpt-4o',
      'openai/o3-mini',
      'google/gemini-2.5-pro',
      'deepseek/deepseek-chat',
      'qwen/qwen3-coder',
    ],
    maxContextLength: {},
    maxOutputTokens: {},
    supportsStreaming: true,
    supportsToolCalling: true,
    supportsSystemMessages: true,
    supportsVision: true,
    supportsAudio: true,
    supportsFineTuning: false,
    supportsEmbeddings: false,
    supportsBatching: true,
    pricing: {},
  };

  private baseUrl: string = DEFAULT_BASE_URL;
  private headers: Record<string, string> = {};
  private cwd: string = process.cwd();

  constructor(options: BaseProviderOptions) {
    super(options);
    // BaseProvider emits an 'error' event on every failed request. Node's
    // EventEmitter throws ERR_UNHANDLED_ERROR when nothing is listening, which
    // would mask the specific error we just threw. This no-op listener keeps
    // the real error as the one the caller sees. It does NOT swallow anything
    // for subscribers: EventEmitter supports many listeners, so a consumer
    // that subscribes still receives the event.
    this.on('error', () => {});
  }

  protected async doInitialize(): Promise<void> {
    if (!this.config.apiKey) {
      throw new AuthenticationError('OpenRouter API key is required', 'openrouter');
    }

    this.baseUrl = (this.config.apiUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
    this.headers = {
      Authorization: `Bearer ${this.config.apiKey}`,
      'Content-Type': 'application/json',
      // OpenRouter convention: identify the integrating app for analytics
      // and rate-limit tiering.
      'HTTP-Referer': 'https://github.com/ruvnet/ruflo',
      'X-Title': 'Ruflo',
    };

    if (typeof this.config.providerOptions?.cwd === 'string') {
      this.cwd = this.config.providerOptions.cwd;
    }
  }

  /** Resolve a logical alias or passthrough slug to a concrete model id. */
  private resolveModel(model?: LLMModel | string): string {
    const raw = (model as string) || (this.config.model as string) || DEFAULT_MODEL;
    return TIER_ALIASES[raw] ?? raw;
  }

  /** Resolve the capability profile for a model id. */
  getCapabilityProfile(modelId: string): ModelCapabilityProfile {
    const overrides = this.config.providerOptions?.modelCapabilities as
      | Record<string, Partial<ModelCapabilityProfile>>
      | undefined;
    return resolveModelCapabilities('openrouter', modelId, {
      overrides,
      cwd: this.cwd,
    });
  }

  /**
   * Reject unsupported requests before any network I/O.
   *
   * Always throws. There is no lenient mode: a silently degraded request is
   * the exact failure this work exists to remove.
   */
  validateRequest(
    request: LLMRequest,
    extras: OpenRouterRequestExtras,
    profile: ModelCapabilityProfile
  ): void {
    const wantsTools = Array.isArray(request.tools) && request.tools.length > 0;
    const wantsSchema = !!extras.responseSchema;
    const wantsReasoning = this.hasReasoningOptions(extras.reasoning);

    // An unverified model asked for advanced features gets the actionable
    // error first — it tells the user to declare capabilities or refresh the
    // catalog, which is the actual fix. This must come before the generic
    // capability errors so the message is useful.
    if (profile.source === 'unverified_default' && (wantsTools || wantsSchema || wantsReasoning)) {
      throw new UnknownModelCapabilityError({
        provider: 'openrouter',
        modelId: profile.modelId,
      });
    }

    if (wantsTools && profile.tools !== 'supported') {
      throw new UnsupportedCapabilityError({
        code: CAPABILITY_CODES.TOOLS,
        provider: 'openrouter',
        modelId: profile.modelId,
        capability: 'tools',
        message:
          `Model "${profile.modelId}" does not declare tool-calling support ` +
          `(capability source: ${profile.source}). Remove the tools, or pick a model that supports them.`,
      });
    }

    if (wantsSchema) {
      if (profile.structuredOutput === 'unsupported') {
        throw new UnsupportedCapabilityError({
          code: CAPABILITY_CODES.STRUCTURED_OUTPUT,
          provider: 'openrouter',
          modelId: profile.modelId,
          capability: 'structuredOutput',
          message:
            `Model "${profile.modelId}" does not declare structured-output support ` +
            `(capability source: ${profile.source}). Remove responseSchema, or pick a model that supports it.`,
        });
      }
      if (profile.structuredOutput === 'tool_strict_fallback') {
        throw new UnsupportedCapabilityError({
          code: CAPABILITY_CODES.STRUCTURED_OUTPUT,
          provider: 'openrouter',
          modelId: profile.modelId,
          capability: 'structuredOutput',
          message:
            `Model "${profile.modelId}" declares only tool_strict_fallback structured output, ` +
            `which this provider does not implement yet. Use a model declaring json_schema or json_object.`,
        });
      }
    }

    if (wantsReasoning) {
      if (profile.reasoningControl === 'unsupported') {
        throw new UnsupportedCapabilityError({
          code: CAPABILITY_CODES.REASONING,
          provider: 'openrouter',
          modelId: profile.modelId,
          capability: 'reasoningControl',
          message:
            `Model "${profile.modelId}" does not declare reasoning control ` +
            `(capability source: ${profile.source}). Remove the reasoning options, or pick a model that supports them.`,
        });
      }
      if (profile.reasoningControl === 'anthropic_thinking') {
        throw new UnsupportedCapabilityError({
          code: CAPABILITY_CODES.REASONING,
          provider: 'openrouter',
          modelId: profile.modelId,
          capability: 'reasoningControl',
          message:
            `Model "${profile.modelId}" expects Anthropic native thinking blocks, which this ` +
            `OpenAI-compatible surface cannot express. Use the direct Anthropic provider.`,
        });
      }
      // The declared mode must match the supplied option shape.
      if (
        profile.reasoningControl === 'openrouter_reasoning_effort' &&
        extras.reasoning?.effort === undefined
      ) {
        throw new UnsupportedParameterError({
          provider: 'openrouter',
          modelId: profile.modelId,
          parameter: 'reasoning.effort',
          message:
            `Model "${profile.modelId}" accepts reasoning by effort only. ` +
            `Supply reasoning.effort ('low' | 'medium' | 'high'), not maxTokens.`,
        });
      }
      if (
        profile.reasoningControl === 'openrouter_reasoning_max_tokens' &&
        extras.reasoning?.maxTokens === undefined
      ) {
        throw new UnsupportedParameterError({
          provider: 'openrouter',
          modelId: profile.modelId,
          parameter: 'reasoning.maxTokens',
          message:
            `Model "${profile.modelId}" accepts reasoning by token budget only. ` +
            `Supply reasoning.maxTokens, not effort.`,
        });
      }
    }

    // Vision / audio content parts.
    for (const msg of request.messages) {
      if (typeof msg.content === 'string') continue;
      for (const part of msg.content) {
        if (part.type === 'image' && profile.vision !== 'supported') {
          throw new UnsupportedCapabilityError({
            code: CAPABILITY_CODES.VISION,
            provider: 'openrouter',
            modelId: profile.modelId,
            capability: 'vision',
            message: `Model "${profile.modelId}" does not declare vision support. Remove the image part or pick a vision-capable model.`,
          });
        }
        if (part.type === 'audio' && profile.audio !== 'supported') {
          throw new UnsupportedCapabilityError({
            code: CAPABILITY_CODES.AUDIO,
            provider: 'openrouter',
            modelId: profile.modelId,
            capability: 'audio',
            message: `Model "${profile.modelId}" does not declare audio support. Remove the audio part or pick an audio-capable model.`,
          });
        }
      }
    }

    // Sampling: an EXPLICIT caller value on a model that rejects it is an
    // error (decision D1). An absent value is simply not injected.
    if (typeof request.temperature === 'number' && !profile.samplingParams.temperature) {
      throw new UnsupportedParameterError({
        provider: 'openrouter',
        modelId: profile.modelId,
        parameter: 'temperature',
      });
    }
    if (typeof request.topP === 'number' && !profile.samplingParams.topP) {
      throw new UnsupportedParameterError({
        provider: 'openrouter',
        modelId: profile.modelId,
        parameter: 'top_p',
      });
    }
    if (typeof request.topK === 'number' && !profile.samplingParams.topK) {
      throw new UnsupportedParameterError({
        provider: 'openrouter',
        modelId: profile.modelId,
        parameter: 'top_k',
      });
    }

    // tool_choice without tools has no meaning on this wire; say so rather
    // than dropping it.
    if (!wantsTools && request.toolChoice !== undefined) {
      throw new UnsupportedParameterError({
        provider: 'openrouter',
        modelId: profile.modelId,
        parameter: 'tool_choice',
        message: 'tool_choice was supplied without any tools. Supply tools, or remove tool_choice.',
      });
    }
  }

  /** True when the caller actually supplied a reasoning option. */
  private hasReasoningOptions(
    reasoning?: { effort?: string; maxTokens?: number }
  ): boolean {
    if (!reasoning) return false;
    return reasoning.effort !== undefined || reasoning.maxTokens !== undefined;
  }

  protected async doComplete(request: LLMRequest): Promise<LLMResponse> {
    const extras = (request.providerOptions ?? {}) as OpenRouterRequestExtras;
    const body = this.buildRequest(request, extras, false);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.timeout || 60000);

    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      const data = (await response.json()) as OpenRouterChatResponse;
      return this.transformResponse(data, request);
    } catch (error) {
      clearTimeout(timeout);
      throw this.transformError(error);
    }
  }

  protected async *doStreamComplete(request: LLMRequest): AsyncIterable<LLMStreamEvent> {
    const extras = (request.providerOptions ?? {}) as OpenRouterRequestExtras;
    const body = this.buildRequest(request, extras, true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), (this.config.timeout || 60000) * 2);

    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }
      if (!response.body) {
        throw new LLMProviderError(
          'OpenRouter returned an empty stream body',
          'OPENROUTER_EMPTY_STREAM',
          'openrouter',
          undefined,
          true
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let promptTokens = 0;
      let completionTokens = 0;
      let collected = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;

          let chunk: OpenRouterChatResponse;
          try {
            chunk = JSON.parse(payload) as OpenRouterChatResponse;
          } catch {
            continue;
          }

          const usage = chunk.usage;
          if (usage?.prompt_tokens) promptTokens = usage.prompt_tokens;
          if (usage?.completion_tokens) completionTokens = usage.completion_tokens;

          // OpenAI-compatible SSE carries the incremental piece on `delta`.
          // Reading `message` here (as an earlier draft did) yields nothing at
          // all — the stream silently produces no content events.
          const delta = chunk.choices?.[0]?.delta;
          if (delta?.content) {
            collected += delta.content;
            yield { type: 'content', delta: { content: delta.content } };
          }
          if (delta?.tool_calls) {
            for (const call of delta.tool_calls) {
              yield {
                type: 'tool_call',
                delta: {
                  toolCall: { id: call.id, type: 'function', function: call.function },
                },
              };
            }
          }
        }
      }

      if (!promptTokens) {
        promptTokens = this.estimateTokens(JSON.stringify(request.messages));
      }
      if (!completionTokens) {
        completionTokens = this.estimateTokens(collected);
      }

      yield {
        type: 'done',
        usage: {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        },
      };
    } catch (error) {
      clearTimeout(timeout);
      throw this.transformError(error);
    } finally {
      clearTimeout(timeout);
    }
  }

  async listModels(): Promise<LLMModel[]> {
    // Authoritative list requires a network call; the seed list keeps this
    // cheap and deterministic for adapter wiring and offline use.
    return this.capabilities.supportedModels;
  }

  async getModelInfo(model: LLMModel): Promise<ModelInfo> {
    const modelId = this.resolveModel(model);
    const profile = this.getCapabilityProfile(modelId);
    const features = ['chat', 'completion'];
    if (profile.tools === 'supported') features.push('tool_calling');
    if (profile.structuredOutput !== 'unsupported') features.push('structured_output');
    if (profile.vision === 'supported') features.push('vision');
    if (profile.audio === 'supported') features.push('audio');
    if (profile.reasoningControl !== 'unsupported') features.push('reasoning');
    if (profile.promptCaching !== 'unsupported') features.push('prompt_caching');

    return {
      model: modelId as LLMModel,
      name: modelId,
      description: `OpenRouter model "${modelId}" (capabilities: ${profile.source})`,
      contextLength: profile.maxContextLength ?? this.capabilities.maxContextLength[modelId] ?? 128000,
      maxOutputTokens: profile.maxOutputTokens ?? this.capabilities.maxOutputTokens[modelId] ?? 4096,
      supportedFeatures: features,
    };
  }

  protected async doHealthCheck(): Promise<HealthCheckResult> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/models`, {
        headers: this.headers,
      });
      return {
        healthy: response.ok,
        timestamp: new Date(),
        ...(response.ok ? {} : { error: `HTTP ${response.status}` }),
      };
    } catch (error) {
      return {
        healthy: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date(),
      };
    }
  }

  /**
   * Build the wire body. Every feature is gated on the declared profile.
   */
  private buildRequest(
    request: LLMRequest,
    extras: OpenRouterRequestExtras,
    stream: boolean
  ): OpenRouterChatRequest {
    const modelId = this.resolveModel(request.model);
    const profile = this.getCapabilityProfile(modelId);

    this.validateRequest(request, extras, profile);

    const body: OpenRouterChatRequest = {
      model: modelId,
      stream,
      messages: this.buildMessages(request, profile),
    };

    if (request.maxTokens || this.config.maxTokens) {
      body.max_tokens = request.maxTokens || this.config.maxTokens;
    }

    // Sampling. An explicit request.* value for a rejecting model already
    // threw in validateRequest(). A config-level default is silently not
    // applied here — documented in bundle/decisions.md as a deliberate
    // choice, because a provider-wide default would otherwise make every
    // rejecting model unusable.
    if (profile.samplingParams.temperature) {
      const temp = request.temperature ?? this.config.temperature;
      if (typeof temp === 'number') body.temperature = temp;
    }
    if (profile.samplingParams.topP) {
      const topP = request.topP ?? this.config.topP;
      if (typeof topP === 'number') body.top_p = topP;
    }
    if (profile.samplingParams.topK) {
      const topK = request.topK ?? this.config.topK;
      if (typeof topK === 'number') body.top_k = topK;
    }

    if (request.stopSequences || this.config.stopSequences) {
      body.stop = request.stopSequences || this.config.stopSequences;
    }

    // Decision D2 — constrain routing only when the body carries parameters
    // that not every upstream serves. Plain text keeps full routing breadth.
    let requiresParameters = false;

    if (request.tools && request.tools.length > 0 && profile.tools === 'supported') {
      body.tools = request.tools.map((tool) => {
        // LLMTool.function is the shared shape and has no `strict` marker, but
        // OpenRouter accepts one. Read it defensively rather than dropping a
        // caller-supplied value.
        const fn = tool.function as typeof tool.function & { strict?: boolean };
        return {
          type: 'function' as const,
          function: {
            name: fn.name,
            description: fn.description ?? '',
            parameters: fn.parameters as Record<string, unknown>,
            ...(fn.strict !== undefined ? { strict: fn.strict } : {}),
          },
        };
      });
      if (request.toolChoice !== undefined) {
        body.tool_choice = request.toolChoice;
      }
      requiresParameters = true;
    }

    if (extras.responseSchema) {
      if (profile.structuredOutput === 'json_object') {
        body.response_format = { type: 'json_object' };
      } else {
        body.response_format = {
          type: 'json_schema',
          json_schema: {
            name: extras.responseSchema.name,
            schema: extras.responseSchema.schema,
            strict: extras.responseSchema.strict !== false,
          },
        };
      }
      requiresParameters = true;
    }

    if (this.hasReasoningOptions(extras.reasoning)) {
      // Send only the shape the model declares — never both.
      body.reasoning =
        profile.reasoningControl === 'openrouter_reasoning_effort'
          ? { effort: extras.reasoning?.effort }
          : { max_tokens: extras.reasoning?.maxTokens };
      requiresParameters = true;
    }

    if (requiresParameters) {
      body.provider = { require_parameters: true };
    }

    return body;
  }

  /**
   * Serialize messages, choosing the caching form the profile declares.
   */
  private buildMessages(
    request: LLMRequest,
    profile: ModelCapabilityProfile
  ): OpenRouterMessage[] {
    const out: OpenRouterMessage[] = [];
    const useMultipartCache = profile.promptCaching === 'openrouter_multipart_ephemeral';
    let breakpointsUsed = 0;

    for (const msg of request.messages) {
      const wantsBreakpoint =
        useMultipartCache &&
        this.config.promptCache !== false &&
        breakpointsUsed < MAX_CACHE_BREAKPOINTS;

      if (typeof msg.content === 'string') {
        if (msg.role === 'system' && wantsBreakpoint) {
          breakpointsUsed += 1;
          out.push({
            role: 'system',
            content: [
              { type: 'text', text: msg.content, cache_control: { type: 'ephemeral' } },
            ] as Array<Record<string, unknown>>,
          });
          continue;
        }
        out.push({
          role: msg.role,
          content: msg.content,
          ...(msg.name ? { name: msg.name } : {}),
          ...(msg.toolCallId ? { tool_call_id: msg.toolCallId } : {}),
          ...(msg.toolCalls ? { tool_calls: msg.toolCalls } : {}),
        });
        continue;
      }

      // Structured content parts → OpenRouter content array. Content is
      // preserved for every role, including system (an earlier draft blanked
      // structured system content).
      const parts = msg.content.map((part) => {
        if (part.type === 'text') {
          return { type: 'text', text: part.text ?? '' };
        }
        if (part.type === 'image') {
          const url =
            part.imageUrl ??
            (part.imageBase64 ? `data:image/jpeg;base64,${part.imageBase64}` : '');
          return { type: 'image_url', image_url: { url } };
        }
        // audio — OpenAI-compatible wire shape is { data, format }, not { url }.
        return { type: 'input_audio', input_audio: this.toAudioPayload(part.audioUrl) };
      });

      const shouldBreakpoint = msg.role === 'system' && wantsBreakpoint;
      if (shouldBreakpoint) {
        breakpointsUsed += 1;
        (parts[0] as Record<string, unknown>).cache_control = { type: 'ephemeral' };
      }

      out.push({
        role: msg.role,
        content: parts as Array<Record<string, unknown>>,
        ...(msg.name ? { name: msg.name } : {}),
        ...(msg.toolCallId ? { tool_call_id: msg.toolCallId } : {}),
        ...(msg.toolCalls ? { tool_calls: msg.toolCalls } : {}),
      });
    }

    return out;
  }

  /** Convert a data URI into the { data, format } audio payload OpenRouter expects. */
  private toAudioPayload(audioUrl?: string): { data: string; format: string } {
    const match = /^data:audio\/([a-z0-9]+);base64,(.*)$/i.exec(audioUrl ?? '');
    if (!match) {
      throw new UnsupportedParameterError({
        provider: 'openrouter',
        modelId: this.resolveModel(),
        parameter: 'audio',
        message:
          'Audio content must be a base64 data URI (data:audio/<format>;base64,...). ' +
          'Remote audio URLs are not supported on the OpenAI-compatible wire.',
      });
    }
    return { data: match[2], format: match[1].toLowerCase() };
  }

  private transformResponse(
    data: OpenRouterChatResponse,
    request: LLMRequest
  ): LLMResponse {
    const choice = data.choices?.[0];
    if (!choice) {
      throw new LLMProviderError(
        'OpenRouter returned no choices',
        'OPENROUTER_NO_CHOICES',
        'openrouter',
        undefined,
        true,
        data
      );
    }

    const modelId = data.model ?? this.resolveModel(request.model);
    const usage = data.usage ?? {};
    const promptTokens = usage.prompt_tokens ?? 0;
    const completionTokens = usage.completion_tokens ?? 0;

    return {
      id: data.id ?? `openrouter-${Date.now()}`,
      model: modelId as LLMModel,
      provider: 'openrouter',
      content: choice.message?.content ?? '',
      toolCalls: choice.message?.tool_calls,
      usage: {
        promptTokens,
        completionTokens,
        totalTokens: usage.total_tokens ?? promptTokens + completionTokens,
      },
      finishReason: this.mapFinishReason(choice.finish_reason),
      metadata: {
        cachedTokens: usage.prompt_tokens_details?.cached_tokens ?? 0,
      },
    };
  }

  private mapFinishReason(
    reason?: string
  ): 'stop' | 'length' | 'tool_calls' | 'content_filter' {
    switch (reason) {
      case 'length':
        return 'length';
      case 'tool_calls':
      case 'function_call':
        return 'tool_calls';
      case 'content_filter':
        return 'content_filter';
      default:
        return 'stop';
    }
  }

  private async handleErrorResponse(response: Response): Promise<never> {
    const errorText = await response.text();
    let parsed: { error?: { message?: string } };
    try {
      parsed = JSON.parse(errorText);
    } catch {
      parsed = { error: { message: errorText } };
    }
    const message = parsed.error?.message ?? 'Unknown error';

    switch (response.status) {
      case 401:
        throw new AuthenticationError(message, 'openrouter', parsed);
      case 404:
        throw new ModelNotFoundError(String(this.config.model), 'openrouter', parsed);
      case 429: {
        const retryAfter = response.headers.get('retry-after');
        throw new RateLimitError(
          message,
          'openrouter',
          retryAfter ? parseInt(retryAfter, 10) : undefined,
          parsed
        );
      }
      default:
        throw new LLMProviderError(
          message,
          `OPENROUTER_${response.status}`,
          'openrouter',
          response.status,
          response.status >= 500,
          parsed
        );
    }
  }
}
