/** Shapes shared with the FastAPI backend. Keep in sync with API_CONTRACT.md. */

/**
 * Legacy top-level status. Kept so the app still works against a backend that
 * has not shipped `decision` yet — see `resolveRepair` in services/repair.ts,
 * which maps it onto a `RepairDecision`.
 */
export type SpeechStatus = 'success' | 'uncertain' | 'retry';

/**
 * How much Revoice trusts its own best prediction.
 *
 * - `high`   speak it immediately, no confirmation
 * - `medium` show the best prediction first, offer alternatives, one tap to speak
 * - `low`    show the best prediction, flagged as uncertain; never auto-speak
 * - `retry`  there was not enough audio/ASR evidence to show anything at all
 */
export type RepairDecision = 'high' | 'medium' | 'low' | 'retry';

/**
 * One complete candidate sentence from the repair model.
 *
 * `confidence` is whatever the model genuinely reports, or null when it does
 * not produce a calibrated score. It is never invented outside DEMO_MODE.
 */
export type RepairCandidate = {
  text: string;
  confidence: number | null;
};

/**
 * A word the recogniser was unsure about.
 *
 * Legacy field. The app no longer builds its interaction around filling this
 * blank — it is only used to synthesise whole-sentence `alternatives` when the
 * backend has not sent any.
 */
export type UncertainWord = {
  /** Zero-based word index inside `raw_transcript`. */
  position: number;
  original: string;
  options: string[];
};

/** Response body of `POST /process-speech`. */
export type ProcessSpeechResponse = {
  status: SpeechStatus;
  raw_transcript: string | null;
  /** The repair model's best complete sentence. */
  repaired_text: string | null;
  confidence: number | null;
  /**
   * Ranked alternative complete sentences, best first, excluding
   * `repaired_text`. Empty when the model offers no runner-up.
   */
  alternatives: RepairCandidate[];
  /**
   * Explicit confidence band. Optional: when the backend omits it the app
   * derives one from `status` and `confidence`.
   */
  decision?: RepairDecision | null;
  /**
   * False when no repair model ran and `repaired_text` is just the raw ASR
   * output passed through. Lets the UI avoid claiming a prediction it did not
   * make. Optional; treated as unknown when absent.
   */
  repair_available?: boolean | null;
  /** Legacy word-level uncertainty. */
  uncertain_words: UncertainWord[];
  /** Absolute URL, or a path like `/audio/result.wav` relative to the API base. */
  audio_url: string | null;
};

/** Screen states of the main interaction flow. */
export type FlowPhase =
  | 'ready'
  | 'recording'
  | 'processing'
  /** Showing Revoice's best prediction for the user to accept or adjust. */
  | 'predicted'
  /** Auto-playing the sentence. */
  | 'speaking'
  /** Playback finished; the sentence stays up with a quick way to speak again. */
  | 'complete'
  | 'retry'
  | 'error';

export type HistoryItem = {
  id: string;
  text: string;
  /** ISO timestamp. */
  createdAt: string;
  audioUrl: string | null;
};
