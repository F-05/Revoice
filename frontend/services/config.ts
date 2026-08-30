/**
 * Runtime configuration.
 *
 * The app talks to the real FastAPI backend by default. DEMO_MODE is kept as a
 * fallback for showing the flow with no backend running — set
 * EXPO_PUBLIC_DEMO_MODE=true in .env to turn it back on.
 */
import { Platform } from 'react-native';

const envFlag = process.env.EXPO_PUBLIC_DEMO_MODE;

export const DEMO_MODE = envFlag === 'true';

/**
 * Base URL of the FastAPI backend.
 *
 * NOTE: `localhost` on a physical phone points at the phone, not your laptop.
 * Set your machine's LAN IP in .env, e.g.
 *   EXPO_PUBLIC_API_URL=http://192.168.1.42:8000
 * The web build can use localhost because it runs on the same machine.
 */
const configured = process.env.EXPO_PUBLIC_API_URL?.trim();

export const API_BASE_URL = (configured || 'http://localhost:8000').replace(/\/+$/, '');

/** Whether the configured URL points at the device itself rather than the server. */
const isLoopback = /^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/i.test(API_BASE_URL);

/**
 * Give up on a slow request rather than leaving the user on a spinner forever.
 * Whisper on CPU takes a few seconds for a short phrase, longer on a cold model.
 */
export const REQUEST_TIMEOUT_MS = 45_000;

/**
 * Development logging. Stripped from production builds by the `__DEV__` guard,
 * so nothing noisy ever reaches a shipped app.
 */
export function devLog(...args: unknown[]) {
  if (__DEV__) console.log('[revoice]', ...args);
}

export function devWarn(...args: unknown[]) {
  if (__DEV__) console.warn('[revoice]', ...args);
}

if (__DEV__) {
  devLog(
    `config: platform=${Platform.OS} demoMode=${DEMO_MODE} apiBaseUrl=${API_BASE_URL}`,
  );
  if (!DEMO_MODE && isLoopback && Platform.OS !== 'web') {
    devWarn(
      `EXPO_PUBLIC_API_URL is ${API_BASE_URL}. The iOS Simulator can reach that, but a ` +
        'physical phone cannot (localhost is the phone) and the Android emulator needs ' +
        'http://10.0.2.2:8000. Set your machine\'s LAN IP in .env for a real device.',
    );
  }
}
