/**
 * React glue for `services/settings.ts`.
 *
 * The service holds the state; this only re-renders the components that care.
 */
import { useEffect, useState } from 'react';
import { getSettings, subscribe, updateSettings, type Settings } from '../services/settings';

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(getSettings);

  useEffect(() => subscribe(setSettings), []);

  return { settings, updateSettings };
}
