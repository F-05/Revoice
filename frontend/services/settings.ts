/**
 * User preferences, kept on the device.
 *
 * Deliberately small: a module-level cache that is hydrated once at startup,
 * plus a subscription so screens re-render when a preference changes. The
 * cache is what the decision layer reads, and it is read synchronously — see
 * the hydration note below.
 *
 * AsyncStorage is already used for history, so nothing new is installed.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = 'revoice.settings.v1';

export type Settings = {
  /**
   * Speak a medium-confidence prediction straight away instead of waiting for
   * the user to tap "Use this".
   *
   * Off by default so existing behaviour is unchanged for new users. It never
   * applies to a `low` prediction — that safety rule lives in `shouldAutoSpeak`
   * in services/repair.ts and is not something this preference can override.
   */
  speakAutomatically: boolean;
};

export const DEFAULT_SETTINGS: Settings = {
  speakAutomatically: false,
};

let current: Settings = DEFAULT_SETTINGS;
let hydrated = false;
let inflight: Promise<Settings> | null = null;

const listeners = new Set<(settings: Settings) => void>();

/**
 * The current preferences, synchronously.
 *
 * Safe to call from a decision path: the root layout does not render the app
 * until `hydrateSettings()` has resolved, so by the time any screen can start
 * a recording this is the stored value rather than the default.
 */
export function getSettings(): Settings {
  return current;
}

/** Whether the stored value has been read yet. */
export function isHydrated(): boolean {
  return hydrated;
}

export function subscribe(listener: (settings: Settings) => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function emit() {
  for (const listener of listeners) listener(current);
}

/**
 * Read preferences from storage into the cache. Idempotent, and concurrent
 * callers share one read.
 */
export async function hydrateSettings(): Promise<Settings> {
  if (hydrated) return current;
  if (inflight) return inflight;

  inflight = (async () => {
    try {
      const stored = await AsyncStorage.getItem(KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Partial<Settings> | null;
        current = {
          speakAutomatically:
            typeof parsed?.speakAutomatically === 'boolean'
              ? parsed.speakAutomatically
              : DEFAULT_SETTINGS.speakAutomatically,
        };
      }
    } catch {
      // Unreadable or corrupt storage falls back to the defaults, which are
      // the conservative choice anyway.
      current = DEFAULT_SETTINGS;
    }
    hydrated = true;
    inflight = null;
    emit();
    return current;
  })();

  return inflight;
}

/**
 * Change one or more preferences.
 *
 * The cache updates first so the switch moves immediately; a storage failure
 * costs the user the preference next launch but never blocks the UI.
 */
export async function updateSettings(patch: Partial<Settings>): Promise<Settings> {
  current = { ...current, ...patch };
  emit();
  try {
    await AsyncStorage.setItem(KEY, JSON.stringify(current));
  } catch {
    // Ignore — same policy as history storage.
  }
  return current;
}
