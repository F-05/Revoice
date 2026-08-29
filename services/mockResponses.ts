/**
 * Mocked backend responses used when DEMO_MODE is on.
 *
 * These match the `POST /process-speech` contract in API_CONTRACT.md exactly,
 * so swapping to the real backend changes nothing above the API layer.
 *
 * IMPORTANT: every `confidence` number in this file is invented for the demo.
 * Nothing here runs in production — `processSpeech` only reads these when
 * DEMO_MODE is on. The real backend must report genuine model scores or null.
 */
import type { ProcessSpeechResponse } from '../types/speech';

export type DemoScenario = 'high' | 'medium' | 'low' | 'retry';

export const DEMO_SCENARIOS: DemoScenario[] = ['high', 'medium', 'low', 'retry'];

export const MOCK_RESPONSES: Record<DemoScenario, ProcessSpeechResponse> = {
  // Confident repair: spoken immediately, no confirmation step.
  high: {
    status: 'success',
    raw_transcript: 'could you get me my glases',
    repaired_text: 'Could you get me my glasses?',
    confidence: 0.91,
    decision: 'high',
    alternatives: [],
    repair_available: true,
    uncertain_words: [],
    // The real backend returns something like '/audio/result.wav'. In demo mode
    // there is no server to fetch it from, so leave it null and let playback
    // use the device voice — the demo should actually speak out loud.
    audio_url: null,
  },

  // Strong best guess with plausible runners-up. The user taps once.
  medium: {
    status: 'uncertain',
    raw_transcript: 'could you bring me my classes',
    repaired_text: 'Could you bring me my glasses?',
    confidence: 0.72,
    decision: 'medium',
    alternatives: [
      { text: 'Could you bring me my glass?', confidence: 0.17 },
      { text: 'Could you bring me my classes?', confidence: 0.11 },
    ],
    repair_available: true,
    uncertain_words: [],
    audio_url: null,
  },

  // Enough survives to show something, not enough to say it out loud.
  low: {
    status: 'uncertain',
    raw_transcript: 'could you bring me the glass is',
    repaired_text: 'Could you bring me the glasses?',
    confidence: 0.38,
    decision: 'low',
    alternatives: [{ text: 'Could you bring me the glass?', confidence: 0.21 }],
    repair_available: true,
    uncertain_words: [],
    audio_url: null,
  },

  // Nothing usable came back at all.
  retry: {
    status: 'retry',
    raw_transcript: null,
    repaired_text: null,
    confidence: 0.12,
    decision: 'retry',
    alternatives: [],
    repair_available: true,
    uncertain_words: [],
    audio_url: null,
  },
};

/**
 * Which mocked response the next recording returns.
 *
 * Defaults to cycling high -> medium -> low -> retry so a single demo run shows
 * every state. The Demo pill on the home screen can pin a specific one.
 */
let pinned: DemoScenario | null = null;
let cursor = 0;

export function pinDemoScenario(scenario: DemoScenario | null) {
  pinned = scenario;
}

export function getPinnedDemoScenario(): DemoScenario | null {
  return pinned;
}

export function nextDemoScenario(): DemoScenario {
  if (pinned) return pinned;
  const scenario = DEMO_SCENARIOS[cursor % DEMO_SCENARIOS.length];
  cursor += 1;
  return scenario;
}
