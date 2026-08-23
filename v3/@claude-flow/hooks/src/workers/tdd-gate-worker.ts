/**
 * TDD Gate Worker — Cold-Reviewer Structural Evidence Enforcement.
 *
 * Port of gate/check-evidence.py. Registered as a PostTask hook.
 * Enforces the 5 structural checks that every implementation round
 * must satisfy before a reviewer can be spawned:
 *
 *   1. evidence/round-N/ directory exists
 *   2. red.txt exists and is non-empty
 *   3. green.txt exists and is non-empty
 *   4. verdict.json exists, parses, has 'verdict' ∈ {approved, changes_requested}, has feedback[]
 *   5. bundle/decisions.md exists
 *   6. manifest.json.prior_reviewers does not contain current REVIEWER_SESSION_ID
 *
 * Gate result codes:
 *   0 — All checks passed
 *   1 — evidence/round-N/ directory not found
 *   2 — red.txt or green.txt missing or empty
 *   3 — verdict.json missing, invalid, or wrong verdict/feedback
 *   4 — bundle/decisions.md missing
 *   5 — manifest.json missing/invalid OR reviewer is a repeat
 *
 * @module tdd-gate-worker
 */

import * as fs from 'fs';
import * as path from 'path';

import {
  HookEvent,
  HookRegistry,
  type HookContext,
  type HookResult,
} from '../types.js';

// ============================================================================
// Types
// ============================================================================

export interface GateResult {
  passed: boolean;
  code: number;
  message: string;
  details: string[];
}

export interface TDDGateConfig {
  workspaceRoot: string;
  maxRounds: number;
  reviewerSessionId?: string;
}

const DEFAULT_CONFIG: TDDGateConfig = {
  workspaceRoot: '/home/wmckenzie/.openclaw/workspace',
  maxRounds: 3,
};

// ============================================================================
// Worker
// ============================================================================

export class TDDGateWorker {
  static readonly HOOK_EVENT = 'post-task';

  private config: TDDGateConfig;

  constructor(config: Partial<TDDGateConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Returns the configured workspace root, reading env-var override if set.
   */
  getWorkspaceRoot(): string {
    return process.env.TDD_WORKSPACE_ROOT || this.config.workspaceRoot;
  }

  // --------------------------------------------------------------------------
  // Core gate check
  // --------------------------------------------------------------------------

  /**
   * Run all 5+1 structural checks against qa-hold/<taskId>/.
   *
   * qa-hold path convention:
   *   {workspaceRoot}/memory-tiers/qa-hold/{taskId}/
   */
  checkEvidence(workspacePath: string, taskId: string, round: number): GateResult {
    const qaHoldDir = path.join(workspacePath, 'memory-tiers', 'qa-hold', taskId);
    const details: string[] = [];

    // 1. evidence/round-N/ directory exists
    const evidenceDir = path.join(qaHoldDir, 'evidence', `round-${round}`);
    if (!this.dirExists(evidenceDir)) {
      return {
        passed: false,
        code: 1,
        message: `evidence/round-${round}/ directory not found at ${evidenceDir}`,
        details,
      };
    }
    details.push(`evidence/round-${round}/ exists`);

    // 2. red.txt exists and is non-empty
    const redPath = path.join(evidenceDir, 'red.txt');
    if (!this.nonEmptyFile(redPath)) {
      return {
        passed: false,
        code: 2,
        message: `red.txt missing or empty at ${redPath}`,
        details,
      };
    }
    details.push('red.txt present and non-empty');

    // 3. green.txt exists and is non-empty
    const greenPath = path.join(evidenceDir, 'green.txt');
    if (!this.nonEmptyFile(greenPath)) {
      return {
        passed: false,
        code: 2,
        message: `green.txt missing or empty at ${greenPath}`,
        details,
      };
    }
    details.push('green.txt present and non-empty');

    // 4. verdict.json exists, parses, has valid 'verdict' field
    const bundleDir = path.join(qaHoldDir, 'bundle');
    const verdictPath = path.join(bundleDir, 'verdict.json');
    const verdictCheck = this.checkVerdict(verdictPath);
    if (!verdictCheck.ok) {
      details.push(verdictCheck.detail);
      return {
        passed: false,
        code: 3,
        message: verdictCheck.message,
        details,
      };
    }
    details.push(verdictCheck.detail);

    // 5. bundle/decisions.md exists
    const decisionsPath = path.join(bundleDir, 'decisions.md');
    if (!this.fileExists(decisionsPath)) {
      return {
        passed: false,
        code: 4,
        message: `bundle/decisions.md missing at ${decisionsPath}`,
        details,
      };
    }
    details.push('bundle/decisions.md exists');

    // 6. manifest.json prior_reviewers check
    const manifestPath = path.join(qaHoldDir, 'manifest.json');
    const manifestCheck = this.checkManifestReviewers(manifestPath);
    if (!manifestCheck.ok) {
      details.push(manifestCheck.detail);
      return {
        passed: false,
        code: 5,
        message: manifestCheck.message,
        details,
      };
    }
    details.push(manifestCheck.detail);

    return {
      passed: true,
      code: 0,
      message: `All 6 structural checks passed for '${taskId}' round ${round}`,
      details,
    };
  }

  // --------------------------------------------------------------------------
  // Individual checks (sync, fs exists-style)
  // --------------------------------------------------------------------------

  private dirExists(dirPath: string): boolean {
    try {
      const stat = fs.statSync(dirPath);
      return stat.isDirectory();
    } catch {
      return false;
    }
  }

  private fileExists(filePath: string): boolean {
    try {
      const stat = fs.statSync(filePath);
      return stat.isFile();
    } catch {
      return false;
    }
  }

  private nonEmptyFile(filePath: string): boolean {
    try {
      const stat = fs.statSync(filePath);
      return stat.isFile() && stat.size > 0;
    } catch {
      return false;
    }
  }

  private readTextFile(filePath: string): string {
    return fs.readFileSync(filePath, 'utf-8').trim();
  }

  // --------------------------------------------------------------------------
  // Verdict check
  // --------------------------------------------------------------------------

  private checkVerdict(verdictPath: string): { ok: boolean; message: string; detail: string } {
    try {
      const raw = this.readTextFile(verdictPath);
      if (!raw) {
        return {
          ok: false,
          message: 'verdict.json is empty',
          detail: 'verdict.json exists but is empty',
        };
      }

      const parsed = JSON.parse(raw);
      const verdict = parsed.verdict;
      const feedback = parsed.feedback;

      if (verdict !== 'approved' && verdict !== 'changes_requested') {
        return {
          ok: false,
          message: `verdict.json has invalid verdict: "${verdict}"`,
          detail: `verdict.json verdict must be "approved" or "changes_requested", got "${verdict}"`,
        };
      }

      if (!Array.isArray(feedback)) {
        return {
          ok: false,
          message: 'verdict.json missing feedback[] array',
          detail: 'verdict.json feedback must be an array',
        };
      }

      return {
        ok: true,
        message: 'verdict.json valid',
        detail: `verdict.json valid (verdict=${verdict}, feedback items=${feedback.length})`,
      };
    } catch (e) {
      const msg = e instanceof SyntaxError ? e.message : String(e);
      return {
        ok: false,
        message: `verdict.json parse failed: ${msg}`,
        detail: `verdict.json not valid JSON: ${msg}`,
      };
    }
  }

  // --------------------------------------------------------------------------
  // Manifest / repeat-reviewer check
  // --------------------------------------------------------------------------

  private checkManifestReviewers(manifestPath: string): { ok: boolean; message: string; detail: string } {
    try {
      const raw = this.readTextFile(manifestPath);
      if (!raw) {
        return {
          ok: false,
          message: 'manifest.json is empty',
          detail: 'manifest.json exists but is empty',
        };
      }

      const parsed = JSON.parse(raw);

      // prior_reviewers is optional — absent means nobody has reviewed yet (fresh run)
      const prior: unknown = parsed.prior_reviewers;
      if (prior === undefined || prior === null) {
        return {
          ok: true,
          message: 'no prior reviewers',
          detail: 'manifest.json: no prior reviewers (fresh task)',
        };
      }

      if (!Array.isArray(prior)) {
        return {
          ok: false,
          message: 'manifest.json prior_reviewers is not an array',
          detail: 'manifest.json prior_reviewers must be an array of reviewer session IDs',
        };
      }

      // If we have a reviewer session ID, check it's not a repeat
      const reviewerId = process.env.REVIEWER_SESSION_ID || this.config.reviewerSessionId;
      if (reviewerId && prior.includes(reviewerId)) {
        return {
          ok: false,
          message: `Reviewer ${reviewerId} has already reviewed this task`,
          detail: `manifest.json prior_reviewers contains current reviewer (${reviewerId})`,
        };
      }

      return {
        ok: true,
        message: `prior reviewers OK (${prior.length} previous)`,
        detail: `manifest.json: ${prior.length} prior reviewer(s), no repeat detected`,
      };
    } catch (e) {
      const msg = e instanceof SyntaxError ? e.message : String(e);
      return {
        ok: false,
        message: `manifest.json parse failed: ${msg}`,
        detail: `manifest.json not valid JSON: ${msg}`,
      };
    }
  }

  // --------------------------------------------------------------------------
  // Hook handler factory
  // --------------------------------------------------------------------------

  /**
   * Returns a `HookHandler` that reads taskId/round from context and runs
   * the structural evidence check. Surfaces failures as HookResult errors.
   */
  createHandler() {
    const worker = this;
    return (context: HookContext): HookResult => {
      const data = (context.data as Record<string, unknown> | undefined) ?? {};
      const taskId = (data.taskId as string) ?? '';
      const round = (data.round as number) ?? 1;

      if (!taskId) {
        return {
          success: true,
          data: { skipped: true, reason: 'no taskId in context.data' },
        };
      }

      const workspaceRoot = worker.getWorkspaceRoot();
      const result = worker.checkEvidence(workspaceRoot, taskId, round);

      if (!result.passed) {
        return {
          success: false,
          data: {
            gateResult: result,
            taskId,
            round,
          },
          errors: [
            `TDD gate failed (code ${result.code}): ${result.message}`,
            ...result.details.map((d) => `  - ${d}`),
          ],
        };
      }

      return {
        success: true,
        data: {
          gateResult: result,
          taskId,
          round,
        },
      };
    };
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createTDDGateWorker(config?: Partial<TDDGateConfig>): TDDGateWorker {
  return new TDDGateWorker(config);
}

// ============================================================================
// Hook registration
// ============================================================================

/**
 * Registers the TDD gate worker on `HookEvent.PostTask`.
 * Returns the hook id for `unregister`/tests.
 * Not auto-registered on the default registry — callers opt in explicitly.
 */
export function registerTDDGateHook(
  registry: HookRegistry,
  config?: Partial<TDDGateConfig>,
): string {
  const worker = createTDDGateWorker(config);
  const handler = worker.createHandler();

  // Dynamic import to avoid circular references
  const { HookPriority } = require('../types.js');

  return registry.register(
    HookEvent.PostTask,
    handler,
    HookPriority.Critical,
    {
      name: 'tdd-gate',
      description: 'TDD gate — cold-reviewer structural evidence enforcement (port of gate/check-evidence.py)',
    },
  );
}