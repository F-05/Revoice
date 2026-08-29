import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Animated, Easing, Platform, StyleSheet, View } from 'react-native';
import { AppHeader, HeaderAction } from '../components/AppHeader';
import { DemoBar } from '../components/DemoBar';
import { Screen } from '../components/Screen';
import { MessageView } from '../components/flow/MessageView';
import { ProcessingView } from '../components/flow/ProcessingView';
import { ReadyView } from '../components/flow/ReadyView';
import { RecordingView } from '../components/flow/RecordingView';
import { SpeakingView } from '../components/flow/SpeakingView';
import { PredictionView } from '../components/flow/PredictionView';
import { spacing } from '../constants/theme';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { useSettings } from '../hooks/useSettings';
import { processSpeech } from '../services/api';
import { DEMO_MODE } from '../services/config';
import { addToHistory } from '../services/history';
import { estimateSpeechDuration, speak, stopSpeaking } from '../services/playback';
import { resolveRepair, shouldAutoSpeak, type RepairOutcome } from '../services/repair';
import type { FlowPhase, ProcessSpeechResponse } from '../types/speech';
import { tidySentence } from '../utils/sentence';

type ErrorKind = 'permission' | 'connection' | null;

/**
 * The conversational loop: speak -> process -> predict -> speak.
 *
 * Revoice predicts first and asks only when it has to. A confident repair is
 * spoken straight away; a less certain one is shown as a complete sentence the
 * user can accept with one tap. Re-recording is a fallback, never the answer
 * to uncertainty.
 */
export default function Home() {
  const router = useRouter();
  const recorder = useAudioRecorder();
  const { settings } = useSettings();

  const [phase, setPhase] = useState<FlowPhase>('ready');
  const [outcome, setOutcome] = useState<RepairOutcome | null>(null);
  const [sentence, setSentence] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [playbackFailed, setPlaybackFailed] = useState(false);
  const [errorKind, setErrorKind] = useState<ErrorKind>(null);

  /** Guards against a second playback starting while one is already running. */
  const playbackId = useRef(0);
  const safetyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearSafetyTimer = useCallback(() => {
    if (safetyTimer.current) {
      clearTimeout(safetyTimer.current);
      safetyTimer.current = null;
    }
  }, []);

  useEffect(
    () => () => {
      clearSafetyTimer();
      stopSpeaking();
    },
    [clearSafetyTimer],
  );

  const tap = useCallback((style: Haptics.ImpactFeedbackStyle) => {
    if (Platform.OS === 'web') return;
    Haptics.impactAsync(style).catch(() => undefined);
  }, []);

  /**
   * Say a sentence out loud and move to `complete` when it finishes.
   * Every entry into the speaking state goes through here.
   */
  const playSentence = useCallback(
    (text: string, url: string | null) => {
      if (!text) return;

      const id = playbackId.current + 1;
      playbackId.current = id;
      const isCurrent = () => playbackId.current === id;

      clearSafetyTimer();
      setPlaybackFailed(false);
      setPhase('speaking');

      const finish = () => {
        if (!isCurrent()) return;
        clearSafetyTimer();
        setPhase('complete');
      };

      speak({
        text,
        audioUrl: url,
        onDone: finish,
        onError: () => {
          if (!isCurrent()) return;
          // Keep the sentence on screen — the user can still read it and retry.
          setPlaybackFailed(true);
          finish();
        },
      }).catch(() => {
        if (!isCurrent()) return;
        setPlaybackFailed(true);
        finish();
      });

      // If a completion event never arrives, don't strand the user on
      // "Speaking…" — move on once the sentence could plausibly be over.
      safetyTimer.current = setTimeout(finish, estimateSpeechDuration(text, Boolean(url)));
    },
    [clearSafetyTimer],
  );

  const beginRecording = useCallback(
    async () => {
      playbackId.current += 1;
      clearSafetyTimer();
      stopSpeaking();
      setPlaybackFailed(false);

      const started = await recorder.start();
      if (!started) {
        setErrorKind('permission');
        setPhase('error');
        return;
      }
      tap(Haptics.ImpactFeedbackStyle.Medium);
      setPhase('recording');
    },
    [clearSafetyTimer, recorder, tap],
  );

  /**
   * Accept a sentence — the prediction, an alternative, or an edit — and say
   * it. There is no second confirmation screen after this point.
   */
  const acceptSentence = useCallback(
    (text: string, url: string | null = null) => {
      const finished = tidySentence(text);
      if (!finished) return;
      setSentence(finished);
      setAudioUrl(url);
      addToHistory({ text: finished, audioUrl: url });
      playSentence(finished, url);
    },
    [playSentence],
  );

  const handleResponse = useCallback(
    (response: ProcessSpeechResponse) => {
      const resolved = resolveRepair(response);
      setOutcome(resolved);

      if (resolved.decision === 'retry') {
        setPhase('retry');
        return;
      }

      // High always speaks; medium speaks only when the user asked it to.
      // `low` can never satisfy this, whatever the preference says.
      if (shouldAutoSpeak(resolved.decision, settings.speakAutomatically)) {
        acceptSentence(resolved.bestText, response.audio_url);
        return;
      }

      // Otherwise show the best complete sentence for one-tap acceptance.
      setPhase('predicted');
    },
    [acceptSentence, settings.speakAutomatically],
  );

  const finishRecording = useCallback(async () => {
    tap(Haptics.ImpactFeedbackStyle.Light);
    setPhase('processing');

    const uri = await recorder.stop();
    if (!uri) {
      setPhase('retry');
      return;
    }

    try {
      const response = await processSpeech(uri);
      handleResponse(response);
    } catch {
      setErrorKind('connection');
      setPhase('error');
    }
  }, [handleResponse, recorder, tap]);

  const resetToReady = useCallback(() => {
    playbackId.current += 1;
    clearSafetyTimer();
    stopSpeaking();
    setPhase('ready');
    setOutcome(null);
    setSentence('');
    setAudioUrl(null);
    setPlaybackFailed(false);
    setErrorKind(null);
  }, [clearSafetyTimer]);

  const isSpeakingPhase = phase === 'speaking' || phase === 'complete';

  return (
    <Screen>
      <AppHeader
        onOpenHistory={phase === 'ready' ? () => router.push('/history') : undefined}
        onOpenSettings={phase === 'ready' ? () => router.push('/settings') : undefined}
        right={
          phase !== 'ready' && phase !== 'processing' && phase !== 'speaking' ? (
            <HeaderAction label="Close" onPress={resetToReady} />
          ) : undefined
        }
      />

      <PhaseFade phase={isSpeakingPhase ? 'speaking' : phase} style={styles.content}>
        {phase === 'ready' ? <ReadyView onStart={() => beginRecording()} /> : null}

        {phase === 'recording' ? (
          <RecordingView
            durationMillis={recorder.durationMillis}
            level={recorder.level}
            onStop={finishRecording}
          />
        ) : null}

        {phase === 'processing' ? <ProcessingView /> : null}

        {isSpeakingPhase ? (
          <SpeakingView
            text={sentence}
            state={phase === 'speaking' ? 'speaking' : 'complete'}
            playbackFailed={playbackFailed}
            onSpeakAgain={() => beginRecording()}
            onReplay={() => playSentence(sentence, audioUrl)}
            onEdit={(next) => {
              const edited = tidySentence(next);
              setSentence(edited);
              // Edited text no longer matches the backend audio.
              setAudioUrl(null);
              playSentence(edited, null);
            }}
          />
        ) : null}

        {phase === 'predicted' && outcome ? (
          <PredictionView
            bestText={outcome.bestText}
            decision={outcome.decision === 'low' ? 'low' : 'medium'}
            alternatives={outcome.alternatives}
            rawTranscript={outcome.rawTranscript}
            wasRepaired={outcome.wasRepaired}
            onAccept={(text) => acceptSentence(text)}
            onTryAgain={beginRecording}
          />
        ) : null}

        {phase === 'retry' ? (
          <MessageView
            heading="I didn't catch that."
            support="That's okay — try saying it again."
            actionLabel="Try again"
            onAction={() => beginRecording()}
            icon="mic"
          />
        ) : null}

        {phase === 'error' ? (
          <MessageView
            heading={
              errorKind === 'permission' ? 'Revoice needs the microphone.' : "I couldn't connect."
            }
            support={
              errorKind === 'permission'
                ? 'Allow microphone access in Settings, then come back and try again.'
                : 'Check your connection, then try again.'
            }
            actionLabel="Try again"
            onAction={() => beginRecording()}
            secondaryLabel="Not now"
            onSecondary={resetToReady}
            icon={errorKind === 'permission' ? 'mic-off' : 'wifi-off'}
            actionIcon={errorKind === 'permission' ? 'mic' : 'refresh-cw'}
          />
        ) : null}
      </PhaseFade>

      {/* Dev-only, kept at the bottom so it stays out of the composition. */}
      {DEMO_MODE && phase === 'ready' ? <DemoBar /> : null}
    </Screen>
  );
}

/**
 * Gentle cross-fade between states. `speaking` and `complete` share a key so
 * finishing playback updates in place instead of re-animating the sentence.
 */
function PhaseFade({
  phase,
  children,
  style,
}: {
  phase: string;
  children: React.ReactNode;
  style?: object;
}) {
  const appear = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    appear.setValue(0);
    Animated.timing(appear, {
      toValue: 1,
      duration: 320,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [appear, phase]);

  return (
    <Animated.View
      style={[
        style,
        {
          opacity: appear,
          transform: [
            { translateY: appear.interpolate({ inputRange: [0, 1], outputRange: [10, 0] }) },
          ],
        },
      ]}>
      <View style={styles.fill}>{children}</View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  content: { flex: 1, marginTop: spacing.sm },
  fill: { flex: 1 },
});
