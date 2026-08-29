/**
 * Microphone recording, wrapped so screens only see:
 * start / stop / duration / live level / permission state.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { devLog, devWarn } from '../services/config';
import {
  RecordingPresets,
  getRecordingPermissionsAsync,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder as useExpoAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';

export type MicPermission = 'unknown' | 'granted' | 'denied';

/** Metering arrives in dBFS (roughly -60 quiet .. 0 loud). Map it to 0..1. */
function meteringToLevel(metering: number | undefined): number {
  if (metering == null || Number.isNaN(metering)) return 0;
  const floor = -55;
  const normalised = (metering - floor) / (0 - floor);
  return Math.min(1, Math.max(0, normalised));
}

export function useAudioRecorder() {
  const recorder = useExpoAudioRecorder({
    ...RecordingPresets.HIGH_QUALITY,
    isMeteringEnabled: true,
  });
  const recorderState = useAudioRecorderState(recorder, 100);

  const [permission, setPermission] = useState<MicPermission>('unknown');
  const [isPreparing, setIsPreparing] = useState(false);
  const isRecordingRef = useRef(false);

  isRecordingRef.current = recorderState.isRecording;

  useEffect(() => {
    let cancelled = false;
    getRecordingPermissionsAsync()
      .then(({ granted, canAskAgain }) => {
        if (cancelled) return;
        setPermission(granted ? 'granted' : canAskAgain ? 'unknown' : 'denied');
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  /** Ask for the microphone. Returns whether we may record. */
  const ensurePermission = useCallback(async (): Promise<boolean> => {
    const current = await getRecordingPermissionsAsync().catch(() => null);
    if (current?.granted) {
      setPermission('granted');
      return true;
    }
    const result = await requestRecordingPermissionsAsync().catch(() => null);
    const granted = Boolean(result?.granted);
    setPermission(granted ? 'granted' : 'denied');
    return granted;
  }, []);

  const start = useCallback(async (): Promise<boolean> => {
    if (isRecordingRef.current) return true;
    const allowed = await ensurePermission();
    if (!allowed) return false;

    setIsPreparing(true);
    try {
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await recorder.prepareToRecordAsync();
      recorder.record();
      devLog('recorder: started');
      return true;
    } catch (error) {
      devWarn('recorder: could not start', error);
      return false;
    } finally {
      setIsPreparing(false);
    }
  }, [ensurePermission, recorder]);

  /** Stop and hand back the local file URI, or null if nothing was captured. */
  const stop = useCallback(async (): Promise<string | null> => {
    try {
      await recorder.stop();
    } catch {
      // Recorder was already stopped or never started.
    }
    await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true }).catch(
      () => undefined,
    );
    const uri = recorder.uri ?? null;
    devLog(`recorder: stopped, uri=${uri ?? 'none'}`);
    return uri;
  }, [recorder]);

  /** Abandon a recording without using the audio. */
  const cancel = useCallback(async () => {
    await stop();
  }, [stop]);

  return {
    start,
    stop,
    cancel,
    ensurePermission,
    permission,
    isPreparing,
    isRecording: recorderState.isRecording,
    durationMillis: recorderState.durationMillis,
    level: meteringToLevel(recorderState.metering),
  };
}
