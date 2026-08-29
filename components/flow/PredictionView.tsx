import { useState } from 'react';
import { Feather } from '@expo/vector-icons';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { Button } from '../Button';
import { SuggestionButton } from '../SuggestionButton';
import { colors, radius, spacing, type } from '../../constants/theme';
import type { RepairCandidate, RepairDecision } from '../../types/speech';

type PredictionViewProps = {
  /** Revoice's best complete sentence. Always shown, never a blank. */
  bestText: string;
  /** `medium` or `low` — `high` is spoken without ever reaching this screen. */
  decision: Extract<RepairDecision, 'medium' | 'low'>;
  alternatives: RepairCandidate[];
  /** What the recogniser heard, shown only when the repair changed it. */
  rawTranscript?: string;
  wasRepaired?: boolean;
  onAccept: (text: string) => void;
  onTryAgain: () => void;
};

/**
 * Revoice's best guess at the whole sentence, offered for one-tap acceptance.
 *
 * The prediction is the subject of this screen. Alternatives are secondary and
 * re-recording is a quiet last resort — the person should not have to say
 * anything twice just because the recogniser was unsure.
 */
export function PredictionView({
  bestText,
  decision,
  alternatives,
  rawTranscript,
  wasRepaired = false,
  onAccept,
  onTryAgain,
}: PredictionViewProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const isEditing = draft !== null;
  const isLow = decision === 'low';

  if (isEditing) {
    return (
      <View style={styles.container}>
        <ScrollView
          contentContainerStyle={styles.body}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}>
          <Text style={styles.heading} accessibilityRole="header">
            Fix the wording
          </Text>
          <TextInput
            value={draft}
            onChangeText={setDraft}
            multiline
            autoFocus
            style={[styles.sentence, styles.input]}
            selectionColor={colors.primary}
            accessibilityLabel="Edit the sentence"
          />
        </ScrollView>
        <View style={styles.actions}>
          <Button
            label="Speak it"
            onPress={() => onAccept(draft.trim() || bestText)}
            icon={<Feather name="volume-2" size={20} color={colors.textOnPrimary} />}
          />
          <TextAction label="Cancel" onPress={() => setDraft(null)} />
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.body}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}>
        {isLow ? (
          <View style={styles.uncertainRow} accessibilityLiveRegion="polite">
            <Feather name="help-circle" size={17} color={colors.textSecondary} />
            <Text style={styles.uncertainText}>I&rsquo;m less sure about this.</Text>
          </View>
        ) : null}

        <Text style={styles.heading} accessibilityRole="header">
          I think you said
        </Text>

        {/* The prediction is the loudest thing on the screen, in both bands. */}
        <Text style={styles.sentence} accessibilityLabel={`I think you said: ${bestText}`}>
          {bestText}
        </Text>

        {wasRepaired && rawTranscript ? (
          <Text style={styles.heard}>I heard &ldquo;{rawTranscript}&rdquo;</Text>
        ) : null}

        {alternatives.length > 0 ? (
          <View style={styles.alternatives}>
            <Text style={styles.alternativesLabel}>Other possibilities</Text>
            {alternatives.map((alternative) => (
              <SuggestionButton
                key={alternative.text}
                label={alternative.text}
                onPress={() => onAccept(alternative.text)}
              />
            ))}
          </View>
        ) : null}
      </ScrollView>

      <View style={styles.actions}>
        <Button
          label="Use this"
          onPress={() => onAccept(bestText)}
          icon={<Feather name="volume-2" size={20} color={colors.textOnPrimary} />}
          accessibilityHint="Speaks this sentence out loud"
        />
        <View style={styles.minorRow}>
          <TextAction label="Edit" onPress={() => setDraft(bestText)} />
          <Text style={styles.dot}>·</Text>
          <TextAction label="Try again" onPress={onTryAgain} />
        </View>
      </View>
    </View>
  );
}

/** Deliberately quiet action — never competes with "Use this". */
function TextAction({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      hitSlop={12}
      style={styles.textAction}>
      <Text style={styles.textActionLabel}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  body: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingVertical: spacing.xl,
  },
  uncertainRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs + 2,
    marginBottom: spacing.md,
  },
  uncertainText: { ...type.label, color: colors.textSecondary },
  heading: { ...type.title, color: colors.text },
  sentence: {
    ...type.speech,
    color: colors.text,
    marginTop: spacing.lg,
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: colors.primary,
    paddingBottom: spacing.sm,
    borderRadius: radius.sm,
  },
  heard: {
    ...type.support,
    color: colors.textSecondary,
    marginTop: spacing.md,
  },
  alternatives: { gap: spacing.md, marginTop: spacing.xl },
  alternativesLabel: { ...type.label, color: colors.textSecondary },
  actions: { gap: spacing.sm },
  minorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  textAction: {
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: spacing.sm,
  },
  textActionLabel: { ...type.support, color: colors.textSecondary },
  dot: { ...type.support, color: colors.border },
});
