/**
 * Turning a backend response into a decision the UI can act on.
 *
 * Revoice predicts first and asks only when it has to. This module decides
 * which of those three things happens, and it is deliberately conservative:
 * when the evidence is thin it lowers its own confidence rather than speaking
 * a sentence the person did not say.
 *
 * No language model runs here. This is scoring and bookkeeping only — the
 * actual repair happens in the backend.
 */
import { rebuildSentence, tidySentence } from '../utils/sentence';
import type {
  ProcessSpeechResponse,
  RepairCandidate,
  RepairDecision,
} from '../types/speech';

/**
 * Confidence at or above this is spoken immediately with no confirmation.
 * Below `MEDIUM_CONFIDENCE` the prediction is shown but never auto-spoken.
 *
 * These are UI thresholds, not a calibrated model. They exist so the app
 * behaves sensibly against a backend that reports only a single number; once
 * the repair model reports its own band in `decision`, that wins.
 */
export const HIGH_CONFIDENCE = 0.75;
export const MEDIUM_CONFIDENCE = 0.45;

/**
 * How much of the raw transcript a repair must keep to be trusted.
 *
 * A repair that shares almost no words with what the recogniser actually heard
 * is not a correction, it is a new sentence. We cannot tell a good rewrite from
 * an invented one, so anything below this is demoted to `low` and shown as a
 * guess rather than spoken. This is the client-side half of "do not use the
 * model as an oracle".
 */
export const MIN_TRANSCRIPT_OVERLAP = 0.5;

/** At most this many alternatives reach the screen — choosing is work. */
export const MAX_ALTERNATIVES = 3;

export type RepairOutcome = {
  decision: RepairDecision;
  /** Revoice's best complete sentence, already tidied for display. */
  bestText: string;
  /** Ranked runners-up, tidied, de-duplicated, capped. */
  alternatives: RepairCandidate[];
  /** What the recogniser heard, kept for the "what I heard" line. */
  rawTranscript: string;
  confidence: number | null;
  /** True when the shown sentence differs from the raw ASR output. */
  wasRepaired: boolean;
  /** Set when a confident-looking repair was demoted for lack of overlap. */
  demotedForOverlap: boolean;
};

function words(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

/**
 * Share of the raw transcript's words that survive into the repair.
 *
 * Returns 1 when there is nothing to compare against, so a missing transcript
 * never triggers the demotion path on its own.
 */
export function transcriptOverlap(rawTranscript: string, repaired: string): number {
  const source = words(rawTranscript);
  if (source.length === 0) return 1;
  const kept = new Set(words(repaired));
  return source.filter((word) => kept.has(word)).length / source.length;
}

/** Sentences the backend offered, cleaned up and stripped of near-duplicates. */
function cleanAlternatives(
  candidates: RepairCandidate[],
  best: string,
): RepairCandidate[] {
  const seen = new Set([best.toLowerCase()]);
  const out: RepairCandidate[] = [];
  for (const candidate of candidates) {
    const text = tidySentence(candidate?.text ?? '');
    if (!text || seen.has(text.toLowerCase())) continue;
    seen.add(text.toLowerCase());
    out.push({
      text,
      confidence: typeof candidate.confidence === 'number' ? candidate.confidence : null,
    });
    if (out.length >= MAX_ALTERNATIVES) break;
  }
  return out;
}

/**
 * Build whole-sentence alternatives from legacy word-level options.
 *
 * A backend that only sends `uncertain_words` still gets the new UI: each
 * candidate word is folded back into the sentence so the user always chooses
 * between complete sentences, never between bare words.
 */
function alternativesFromUncertainWords(
  response: ProcessSpeechResponse,
): RepairCandidate[] {
  const uncertain = response.uncertain_words?.[0];
  const raw = response.raw_transcript ?? '';
  if (!uncertain || !raw.trim()) return [];
  return uncertain.options.map((option) => ({
    text: rebuildSentence(raw, uncertain, option),
    // The legacy contract carries no per-option score. Say so rather than
    // making one up.
    confidence: null,
  }));
}

/** Map the legacy `status` + `confidence` pair onto a band. */
function deriveDecision(response: ProcessSpeechResponse, hasText: boolean): RepairDecision {
  if (!hasText) return 'retry';
  if (response.status === 'retry') return 'retry';

  const confidence = response.confidence;
  if (typeof confidence !== 'number') {
    // No score at all: trust `status`, and never auto-speak on a bare
    // "uncertain".
    return response.status === 'success' ? 'high' : 'medium';
  }
  if (response.status === 'success' && confidence >= HIGH_CONFIDENCE) return 'high';
  if (confidence >= MEDIUM_CONFIDENCE) return 'medium';
  return 'low';
}

/**
 * Whether a resolved decision may be spoken without asking the user first.
 *
 * This is post-decision policy only: it reads the band, it never influences
 * which band a repair receives. `high` always speaks. `medium` speaks only
 * when the user has turned "Speak automatically" on.
 *
 * `low` and `retry` never qualify, whatever the preference says. That is the
 * point at which Revoice does not have the evidence to speak on someone's
 * behalf, and no setting is allowed to override it.
 */
export function shouldAutoSpeak(
  decision: RepairDecision,
  speakAutomatically: boolean,
): boolean {
  if (decision === 'high') return true;
  if (decision === 'medium') return speakAutomatically;
  return false;
}

/**
 * Decide what the UI should do with one backend response.
 *
 * The order matters: the backend's explicit `decision` is honoured first, then
 * the overlap guard can only ever lower it, never raise it.
 */
export function resolveRepair(response: ProcessSpeechResponse): RepairOutcome {
  const rawTranscript = (response.raw_transcript ?? '').trim();
  // Suggestion-first backend: `repaired_text` stays the verbatim ASR sentence
  // and the selector's preferred hypothesis arrives in `suggested_text`. The
  // suggestion becomes the prediction the user confirms with one tap, and the
  // ASR sentence is kept as the first alternative so "keep what I said" is
  // always one tap away too. Nothing is ever auto-spoken on this path: the
  // backend marks these responses `uncertain`, which maps to medium/low.
  const suggestion = (response.suggested_text ?? '').trim();
  const bestRaw = suggestion || (response.repaired_text ?? response.raw_transcript ?? '').trim();
  const bestText = tidySentence(bestRaw);
  const hasText = bestText.length > 0;

  let decision: RepairDecision =
    response.decision && response.decision !== 'retry' && hasText
      ? response.decision
      : deriveDecision(response, hasText);

  const supplied = Array.isArray(response.alternatives) ? response.alternatives : [];
  const withKeepOriginal = suggestion && response.repaired_text
    ? [{ text: response.repaired_text, confidence: null }, ...supplied]
    : supplied;
  const alternatives = cleanAlternatives(
    withKeepOriginal.length > 0 ? withKeepOriginal : alternativesFromUncertainWords(response),
    bestText,
  );

  const wasRepaired =
    hasText && rawTranscript.length > 0 && tidySentence(rawTranscript) !== bestText;

  // Conservative guard: a "repair" that keeps almost none of what was heard is
  // treated as a guess, whatever the backend claimed.
  let demotedForOverlap = false;
  if (
    hasText &&
    wasRepaired &&
    decision !== 'low' &&
    decision !== 'retry' &&
    transcriptOverlap(rawTranscript, bestText) < MIN_TRANSCRIPT_OVERLAP
  ) {
    decision = 'low';
    demotedForOverlap = true;
  }

  return {
    decision,
    bestText,
    alternatives,
    rawTranscript,
    confidence: typeof response.confidence === 'number' ? response.confidence : null,
    wasRepaired,
    demotedForOverlap,
  };
}
