/**
 * Recent clarified phrases, kept on the device only.
 * No backend, no database — AsyncStorage is enough for now.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { HistoryItem } from '../types/speech';

const KEY = 'revoice.history.v1';
const MAX_ITEMS = 30;

/** Shown on a fresh install so the Recent screen is never a dead end. */
const SEED: HistoryItem[] = [
  { id: 'seed-1', text: 'Could you get me some water?', createdAt: '', audioUrl: null },
  { id: 'seed-2', text: 'Please close the door.', createdAt: '', audioUrl: null },
  { id: 'seed-3', text: "I'd like to go outside.", createdAt: '', audioUrl: null },
];

export async function loadHistory(): Promise<HistoryItem[]> {
  try {
    const stored = await AsyncStorage.getItem(KEY);
    if (!stored) return SEED;
    const parsed = JSON.parse(stored) as HistoryItem[];
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : SEED;
  } catch {
    return SEED;
  }
}

export async function addToHistory(
  item: Omit<HistoryItem, 'id' | 'createdAt'>,
): Promise<HistoryItem[]> {
  const entry: HistoryItem = {
    ...item,
    id: `${Date.now()}`,
    createdAt: new Date().toISOString(),
  };
  const existing = await loadHistory();
  // Drop the seeded examples once the user has said something real.
  const real = existing.filter((candidate) => !candidate.id.startsWith('seed-'));
  const next = [entry, ...real].slice(0, MAX_ITEMS);
  try {
    await AsyncStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Storage failures should never break the main flow.
  }
  return next;
}

export async function clearHistory(): Promise<void> {
  try {
    await AsyncStorage.removeItem(KEY);
  } catch {
    // Ignore.
  }
}

const ONBOARDING_KEY = 'revoice.onboarded.v1';

export async function hasSeenOnboarding(): Promise<boolean> {
  try {
    return (await AsyncStorage.getItem(ONBOARDING_KEY)) === 'true';
  } catch {
    return false;
  }
}

export async function markOnboardingSeen(): Promise<void> {
  try {
    await AsyncStorage.setItem(ONBOARDING_KEY, 'true');
  } catch {
    // Ignore — worst case the user sees onboarding again.
  }
}
