/**
 * Playing the clarified sentence out loud.
 *
 * Preferred path: play the audio file the backend returned (`audio_url`).
 * Fallback: the device's built-in voice, so the sentence is still spoken in
 * DEMO_MODE, when the backend returns no audio, and when a returned file fails
 * to load. No TTS is implemented here — that stays on the backend; this is only
 * the OS voice used as a safety net.
 */
import { createAudioPlayer, setAudioModeAsync } from 'expo-audio';
import * as Speech from 'expo-speech';
import { resolveAudioUrl } from './api';
import { devLog, devWarn } from './config';

export type SpeakOptions = {
  text: string;
  audioUrl?: string | null;
  /** Playback finished on its own. */
  onDone?: () => void;
  /** Nothing could be played at all. */
  onError?: (error: unknown) => void;
};

/** How long we wait for backend audio to start before using the device voice. */
export const START_TIMEOUT_MS = 5000;

let activePlayer: ReturnType<typeof createAudioPlayer> | null = null;
let watchdog: ReturnType<typeof setTimeout> | null = null;

/**
 * Incremented every time playback is started or stopped. Callbacks compare
 * against the value they captured, so a superseded playback can never move the
 * UI on — this is what prevents duplicate or stale "finished" events.
 */
let currentToken = 0;

function clearWatchdog() {
  if (watchdog) {
    clearTimeout(watchdog);
    watchdog = null;
  }
}

function releasePlayer() {
  if (!activePlayer) return;
  try {
    activePlayer.remove();
  } catch {
    // Already released.
  }
  activePlayer = null;
}

export function stopSpeaking() {
  currentToken += 1;
  clearWatchdog();
  Speech.stop();
  releasePlayer();
}

function speakWithDevice(options: SpeakOptions, isCurrent: () => boolean) {
  try {
    Speech.speak(options.text, {
      rate: 0.95,
      pitch: 1.0,
      onDone: () => {
        if (isCurrent()) options.onDone?.();
      },
      // `onStopped` fires when we interrupt on purpose — not a completion.
      onError: (error) => {
        if (isCurrent()) options.onError?.(error);
      },
    });
  } catch (error) {
    if (isCurrent()) options.onError?.(error);
  }
}

/**
 * Speak a sentence. Resolves once playback has been handed off — use `onDone`
 * for completion and `onError` for "nothing could be played".
 */
export async function speak(options: SpeakOptions): Promise<void> {
  stopSpeaking();
  const token = currentToken;
  const isCurrent = () => token === currentToken;

  await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: false }).catch(
    () => undefined,
  );
  if (!isCurrent()) return;

  const url = resolveAudioUrl(options.audioUrl);
  if (!url) {
    devLog('playback: no backend audio — using the device voice.');
    speakWithDevice(options, isCurrent);
    return;
  }

  devLog(`playback: ${url}`);

  try {
    const player = createAudioPlayer({ uri: url });
    activePlayer = player;
    let started = false;

    player.addListener('playbackStatusUpdate', (status) => {
      if (!isCurrent()) return;
      if (status.playing && !started) {
        started = true;
        clearWatchdog();
        devLog('playback: backend audio started.');
      }
      if (status.didJustFinish) {
        clearWatchdog();
        releasePlayer();
        options.onDone?.();
      }
    });

    // If the file never starts (missing, unreachable, wrong format) the user
    // still hears their sentence rather than silence.
    watchdog = setTimeout(() => {
      if (!isCurrent() || started) return;
      devWarn(`playback: ${url} did not start in ${START_TIMEOUT_MS}ms — device voice instead.`);
      releasePlayer();
      speakWithDevice(options, isCurrent);
    }, START_TIMEOUT_MS);

    player.play();
  } catch (error) {
    devWarn(`playback: could not open ${url} — device voice instead.`, error);
    releasePlayer();
    speakWithDevice(options, isCurrent);
  }
}

/**
 * Rough upper bound on how long a sentence takes to say, used for a safety
 * timer so the UI never sticks on "Speaking…" if no completion event arrives.
 * When backend audio is involved, allow for the watchdog falling back to the
 * device voice first — otherwise the safety timer would cut that off.
 */
export function estimateSpeechDuration(text: string, hasAudioUrl = false): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return 2500 + words * 450 + (hasAudioUrl ? START_TIMEOUT_MS : 0);
}
