import { Feather } from '@expo/vector-icons';
import { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { AudioWaveform } from '../AudioWaveform';
import { Button } from '../Button';
import { colors, radius, spacing, type } from '../../constants/theme';

type SpeakingViewProps = {
  text: string;
  /** `speaking` while audio plays, `complete` once it has finished. */
  state: 'speaking' | 'complete';
  /** Set when nothing could be played — the sentence still stays visible. */
  playbackFailed?: boolean;
  onSpeakAgain: () => void;
  onReplay: () => void;
  onEdit: (next: string) => void;
};

/**
 * The clarified sentence, spoken automatically. There is no "Sounds right?"
 * step — Revoice speaks when it is confident, and the only prominent action
 * afterwards is starting the next sentence.
 */
export function SpeakingView({
  text,
  state,
  playbackFailed = false,
  onSpeakAgain,
  onReplay,
  onEdit,
}: SpeakingViewProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const isEditing = draft !== null;
  const isSpeaking = state === 'speaking';

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.body}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}>
        <View style={styles.status} accessibilityLiveRegion="polite">
          {isSpeaking ? (
            <>
              <Feather name="volume-2" size={17} color={colors.primary} />
              <Text style={[styles.statusText, { color: colors.primary }]}>Speaking…</Text>
            </>
          ) : (
            <>
              <Feather name="check" size={17} color={colors.success} />
              <Text style={[styles.statusText, { color: colors.success }]}>Spoken</Text>
            </>
          )}
        </View>

        {isEditing ? (
          <TextInput
            value={draft}
            onChangeText={setDraft}
            multiline
            autoFocus
            style={[styles.sentence, styles.input]}
            selectionColor={colors.primary}
            accessibilityLabel="Edit the sentence"
          />
        ) : (
          <Text style={styles.sentence}>{text}</Text>
        )}

        {isSpeaking && !isEditing ? (
          <View style={styles.waveform}>
            <AudioWaveform level={0.55} active height={44} barCount={17} />
          </View>
        ) : null}

        {playbackFailed && !isSpeaking ? (
          <Text style={styles.notice}>I couldn&rsquo;t play that out loud.</Text>
        ) : null}
      </ScrollView>

      {isEditing ? (
        <View style={styles.actions}>
          <Button
            label="Speak it"
            onPress={() => {
              onEdit(draft.trim() || text);
              setDraft(null);
            }}
            icon={<Feather name="volume-2" size={20} color={colors.textOnPrimary} />}
          />
          <TextAction label="Cancel" onPress={() => setDraft(null)} />
        </View>
      ) : state === 'complete' ? (
        <View style={styles.actions}>
          <Button
            label="Speak again"
            onPress={onSpeakAgain}
            icon={<Feather name="mic" size={20} color={colors.textOnPrimary} />}
            accessibilityHint="Records your next sentence"
          />
          <View style={styles.minorRow}>
            <TextAction label="Replay" onPress={onReplay} />
            <Text style={styles.dot}>·</Text>
            <TextAction label="Edit" onPress={() => setDraft(text)} />
          </View>
        </View>
      ) : null}
    </View>
  );
}

/** Deliberately quiet action — never competes with "Speak again". */
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
  status: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs + 2,
    marginBottom: spacing.md,
  },
  statusText: { ...type.label },
  sentence: {
    ...type.speech,
    color: colors.text,
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: colors.primary,
    paddingBottom: spacing.sm,
    borderRadius: radius.sm,
  },
  waveform: { marginTop: spacing.xl, alignItems: 'flex-start' },
  notice: {
    ...type.support,
    color: colors.textSecondary,
    marginTop: spacing.lg,
  },
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
  textActionLabel: {
    ...type.support,
    color: colors.textSecondary,
  },
  dot: { ...type.support, color: colors.border },
});
