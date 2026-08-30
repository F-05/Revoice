import { StyleSheet, Text, View } from 'react-native';
import { RecordButton } from '../RecordButton';
import { colors, spacing, type } from '../../constants/theme';

type ReadyViewProps = {
  onStart: () => void;
};

/**
 * Deliberate vertical rhythm rather than one big centred block: heading sits
 * in the upper-middle, the microphone in the lower-middle, and the space
 * between them is left empty on purpose.
 */
export function ReadyView({ onStart }: ReadyViewProps) {
  return (
    <View style={styles.container}>
      <View style={styles.spacerTop} />

      <View style={styles.copy}>
        <Text style={styles.heading} accessibilityRole="header">
          Ready when you are
        </Text>
        <Text style={styles.support}>Tap the microphone and speak naturally.</Text>
      </View>

      <View style={styles.spacerMiddle} />

      <View style={styles.micBlock}>
        <RecordButton isRecording={false} onPress={onStart} />
        <Text style={styles.hint}>Tap to speak</Text>
      </View>

      <View style={styles.spacerBottom} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center' },
  spacerTop: { flex: 0.85 },
  spacerMiddle: { flex: 1.1 },
  spacerBottom: { flex: 0.7 },
  copy: { alignItems: 'center', maxWidth: 300 },
  heading: {
    ...type.title,
    color: colors.text,
    textAlign: 'center',
  },
  support: {
    ...type.support,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  micBlock: { alignItems: 'center' },
  hint: {
    ...type.label,
    color: colors.textSecondary,
    marginTop: spacing.md,
  },
});
