/**
 * Capability registry — declared capabilities, never inferred.
 *
 * The point of these tests: a model's features come from an explicit record
 * (config override, catalog cache, or curated manifest), never from a regex
 * over its name. Unknown models must degrade to a conservative profile and
 * say so, so callers can fail loudly instead of guessing.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import {
  resolveModelCapabilities,
  unverifiedDefault,
  isVerifiedProfile,
  loadOpenRouterModelCatalogCache,
  STATIC_CAPABILITY_MANIFEST,
} from '../capability-registry.js';

describe('resolveModelCapabilities', () => {
  let tmp: string;

  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), 'ruflo-caps-'));
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it('returns a conservative unverified profile for an unknown model', () => {
    const profile = resolveModelCapabilities('openrouter', 'nobody/never-heard-of-it', {
      cwd: tmp,
      skipCatalog: true,
    });

    expect(profile.source).toBe('unverified_default');
    expect(profile.tools).toBe('unsupported');
    expect(profile.structuredOutput).toBe('unsupported');
    expect(profile.reasoningControl).toBe('unsupported');
    expect(isVerifiedProfile(profile)).toBe(false);
  });

  it('resolves a known model from the static manifest', () => {
    const profile = resolveModelCapabilities('openrouter', 'openai/gpt-4.1', {
      cwd: tmp,
      skipCatalog: true,
    });

    expect(profile.source).toBe('static_manifest');
    expect(profile.tools).toBe('supported');
    expect(profile.structuredOutput).toBe('json_schema');
    expect(isVerifiedProfile(profile)).toBe(true);
  });

  it('declares sampling rejection for the adaptive-thinking Anthropic family', () => {
    const profile = resolveModelCapabilities('openrouter', 'anthropic/claude-opus-4-8', {
      cwd: tmp,
      skipCatalog: true,
    });

    expect(profile.samplingParams.temperature).toBe(false);
    expect(profile.samplingParams.topP).toBe(false);
  });

  it('accepts sampling params for a model that does not reject them', () => {
    const profile = resolveModelCapabilities('openrouter', 'openai/gpt-4.1', {
      cwd: tmp,
      skipCatalog: true,
    });

    expect(profile.samplingParams.temperature).toBe(true);
  });

  it('lets an explicit config override win over everything else', () => {
    const profile = resolveModelCapabilities('openrouter', 'openai/gpt-4.1', {
      cwd: tmp,
      skipCatalog: true,
      overrides: {
        'openai/gpt-4.1': { tools: 'unsupported', vision: 'unsupported' },
      },
    });

    expect(profile.source).toBe('config_override');
    expect(profile.tools).toBe('unsupported');
    expect(profile.vision).toBe('unsupported');
  });

  it('reads capabilities from the cached OpenRouter catalog when present', () => {
    const cacheDir = join(tmp, '.claude-flow', 'cache');
    mkdirSync(cacheDir, { recursive: true });
    writeFileSync(
      join(cacheDir, 'openrouter-models.json'),
      JSON.stringify({
        'vendor/brand-new-model': {
          supportedParameters: ['tools', 'tool_choice', 'structured_outputs', 'temperature'],
          contextLength: 200000,
          maxOutputTokens: 16384,
          inputModalities: ['text', 'image'],
        },
      })
    );

    expect(loadOpenRouterModelCatalogCache(tmp)).toBeDefined();

    const profile = resolveModelCapabilities('openrouter', 'vendor/brand-new-model', {
      cwd: tmp,
    });

    expect(profile.source).toBe('openrouter_catalog');
    expect(profile.tools).toBe('supported');
    expect(profile.structuredOutput).toBe('json_schema');
    expect(profile.vision).toBe('supported');
    expect(profile.audio).toBe('unsupported');
    expect(profile.maxContextLength).toBe(200000);
  });

  it('catalog wins over the static manifest for the same model id', () => {
    const cacheDir = join(tmp, '.claude-flow', 'cache');
    mkdirSync(cacheDir, { recursive: true });
    writeFileSync(
      join(cacheDir, 'openrouter-models.json'),
      JSON.stringify({
        'openai/gpt-4.1': { supportedParameters: ['temperature'], inputModalities: ['text'] },
      })
    );

    const profile = resolveModelCapabilities('openrouter', 'openai/gpt-4.1', { cwd: tmp });
    expect(profile.source).toBe('openrouter_catalog');
    expect(profile.tools).toBe('unsupported');
  });

  it('survives a corrupt catalog cache without throwing', () => {
    const cacheDir = join(tmp, '.claude-flow', 'cache');
    mkdirSync(cacheDir, { recursive: true });
    writeFileSync(join(cacheDir, 'openrouter-models.json'), '{not valid json');

    expect(loadOpenRouterModelCatalogCache(tmp)).toBeUndefined();

    const profile = resolveModelCapabilities('openrouter', 'openai/gpt-4.1', { cwd: tmp });
    expect(profile.source).toBe('static_manifest');
  });

  it('gives direct Anthropic its own profile shape with the managed-agents service', () => {
    const profile = resolveModelCapabilities('anthropic', 'claude-sonnet-4-6');
    expect(profile.promptCaching).toBe('anthropic_ephemeral');
    expect(profile.reasoningControl).toBe('anthropic_thinking');
    expect(profile.managedAgentsService).toBe('supported');
  });

  it('never grants the managed-agents service to an OpenRouter model', () => {
    for (const modelId of Object.keys(STATIC_CAPABILITY_MANIFEST)) {
      const profile = resolveModelCapabilities('openrouter', modelId, { skipCatalog: true });
      expect(profile.managedAgentsService).toBe('unsupported');
    }
  });

  it('unverifiedDefault is conservative across every advanced capability', () => {
    const p = unverifiedDefault('openrouter', 'anything');
    expect(p.tools).toBe('unsupported');
    expect(p.structuredOutput).toBe('unsupported');
    expect(p.promptCaching).toBe('unsupported');
    expect(p.reasoningControl).toBe('unsupported');
    expect(p.audio).toBe('unsupported');
    expect(p.managedAgentsService).toBe('unsupported');
  });
});
