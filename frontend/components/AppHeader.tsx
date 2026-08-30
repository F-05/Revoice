import { Feather } from '@expo/vector-icons';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { BrandMark } from './BrandMark';
import { colors, fonts, radius, spacing, type } from '../constants/theme';

type AppHeaderProps = {
  onOpenHistory?: () => void;
  onOpenSettings?: () => void;
  /** Right-hand slot replacement, e.g. a Close action. */
  right?: React.ReactNode;
};

/** A quiet text action for the header's right slot. */
export function HeaderAction({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      hitSlop={12}
      style={({ pressed }) => [styles.headerAction, pressed && { opacity: 0.6 }]}>
      <Text style={styles.headerActionLabel}>{label}</Text>
    </Pressable>
  );
}

/** Light-touch branding, plus small pills into Recent and Settings. Never dominant. */
export function AppHeader({ onOpenHistory, onOpenSettings, right }: AppHeaderProps) {
  return (
    <View style={styles.header}>
      <View style={styles.brand}>
        <BrandMark size={28} />
        <Text style={styles.wordmark}>Revoice</Text>
      </View>
      {right ?? (
        <View style={styles.actions}>
          {onOpenHistory ? (
            <Pressable
              onPress={onOpenHistory}
              accessibilityRole="button"
              accessibilityLabel="Recent phrases"
              accessibilityHint="Shows sentences Revoice has made clear before"
              hitSlop={10}
              style={({ pressed }) => [styles.pill, pressed && styles.pillPressed]}>
              <Feather name="clock" size={14} color={colors.textSecondary} />
              <Text style={styles.pillLabel}>Recent</Text>
            </Pressable>
          ) : null}
          {onOpenSettings ? (
            <Pressable
              onPress={onOpenSettings}
              accessibilityRole="button"
              accessibilityLabel="Settings"
              hitSlop={10}
              style={({ pressed }) => [styles.iconPill, pressed && styles.pillPressed]}>
              <Feather name="settings" size={16} color={colors.textSecondary} />
            </Pressable>
          ) : null}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 40,
  },
  brand: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  actions: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  wordmark: {
    fontFamily: fonts.semibold,
    fontSize: 16,
    lineHeight: 21,
    letterSpacing: -0.2,
    color: colors.text,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs + 1,
    paddingVertical: spacing.sm - 1,
    paddingHorizontal: spacing.md - 2,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  /** Same pill, sized for a lone icon. */
  iconPill: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 36,
    height: 36,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  pillPressed: { backgroundColor: colors.surfaceMuted },
  pillLabel: {
    ...type.caption,
    color: colors.textSecondary,
  },
  headerAction: {
    minHeight: 40,
    justifyContent: 'center',
    paddingHorizontal: spacing.sm,
  },
  headerActionLabel: {
    ...type.label,
    color: colors.textSecondary,
  },
});
