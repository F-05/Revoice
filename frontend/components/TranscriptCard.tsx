import { StyleSheet, Text, TextInput, View } from 'react-native';
import { colors, radius, spacing, type } from '../constants/theme';

type TranscriptCardProps = {
  /** Small label above the sentence, e.g. "I heard:". */
  label: string;
  text: string;
  editable?: boolean;
  onChangeText?: (next: string) => void;
};

/**
 * Shows the clarified sentence. This is the payload of the whole app, so it
 * gets the largest type on the screen and plenty of room around it.
 */
export function TranscriptCard({
  label,
  text,
  editable = false,
  onChangeText,
}: TranscriptCardProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.label} accessibilityRole="header">
        {label}
      </Text>
      {editable ? (
        <TextInput
          value={text}
          onChangeText={onChangeText}
          multiline
          autoFocus
          style={[styles.speech, styles.input]}
          accessibilityLabel="Edit the sentence"
          selectionColor={colors.primary}
        />
      ) : (
        <Text style={styles.speech} accessibilityLabel={`I heard: ${text}`}>
          {text}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: '100%' },
  label: {
    ...type.label,
    color: colors.textSecondary,
    marginBottom: spacing.md,
  },
  speech: {
    ...type.speech,
    color: colors.text,
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: colors.primary,
    paddingBottom: spacing.sm,
    borderRadius: radius.sm,
  },
});
