/**
 * Small text helpers for tidying and rebuilding sentences.
 * Deliberately dumb string work — no language model here.
 */
import type { UncertainWord } from '../types/speech';

const QUESTION_STARTERS = [
  'could',
  'can',
  'would',
  'will',
  'do',
  'does',
  'did',
  'is',
  'are',
  'am',
  'may',
  'might',
  'should',
  'who',
  'what',
  'where',
  'when',
  'why',
  'how',
];

/** Capitalise, and finish with `?` or `.` if the speaker did not. */
export function tidySentence(input: string): string {
  const trimmed = input.trim().replace(/\s+/g, ' ');
  if (!trimmed) return '';
  const capitalised = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
  if (/[.?!]$/.test(capitalised)) return capitalised;
  const firstWord = capitalised.split(' ')[0].toLowerCase().replace(/[^a-z']/g, '');
  return `${capitalised}${QUESTION_STARTERS.includes(firstWord) ? '?' : '.'}`;
}

/** Replace the uncertain word with `replacement` and tidy the result. */
export function rebuildSentence(
  rawTranscript: string,
  uncertain: UncertainWord | undefined,
  replacement: string,
): string {
  const words = rawTranscript.trim().split(/\s+/).filter(Boolean);
  const index =
    uncertain && uncertain.position >= 0 && uncertain.position < words.length
      ? uncertain.position
      : words.findIndex((word) => word === uncertain?.original);

  if (index >= 0) {
    words[index] = replacement;
  } else {
    words.push(replacement);
  }
  return tidySentence(words.join(' '));
}

/** `07` -> `0:07` style timer text. */
export function formatDuration(millis: number): string {
  const totalSeconds = Math.max(0, Math.floor(millis / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
