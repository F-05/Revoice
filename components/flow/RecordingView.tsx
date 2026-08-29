import { StyleSheet, Text, View } from 'react-native';
import { AudioWaveform } from '../AudioWaveform';
import { RecordButton } from '../RecordButton';
import { colors, radius, spacing, type } from '../../constants/theme';
import { formatDuration } from '../../utils/sentence';

type RecordingViewProps = {
  durationMillis: number;
  level: number;
  onStop: () => void;
};

export function RecordingView({ durationMillis, level, onStop }: RecordingViewProps) {
  return (
    <View style={styles.container}>
      <View style={styles.copy}>
        <Text style={styles.heading} accessibilityRole="header">
          Listening…
        </Text>
        <Text style={styles.support}>Speak naturally. Take your time.</Text>
      </View>

      <View style={styles.timerPill}>
        <View style={styles.dot} />
        <Text style={styles.timer} accessibilityLabel={`Recording, ${Math.round(durationMillis / 1000)} seconds`}>
          {formatDuration(durationMillis)}
        </Text>
      </View>

      <AudioWaveform level={level} active />

      <RecordButton isRecording onPress={onStop} size={148} />

      <Text style={styles.hint}>Tap to stop</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  copy: { alignItems: 'center' },
  heading: { ...type.title, color: colors.text, textAlign: 'center' },
  support: {
    ...type.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  timerPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.lg,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceMuted,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
  },
  timer: { ...type.label, color: colors.text, fontVariant: ['tabular-nums'] },
  hint: { ...type.label, color: colors.textSecondary, marginTop: spacing.md },
});
