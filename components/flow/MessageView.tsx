import { Feather } from '@expo/vector-icons';
import { StyleSheet, Text, View } from 'react-native';
import { Button } from '../Button';
import { colors, radius, spacing, type } from '../../constants/theme';

type MessageViewProps = {
  heading: string;
  support: string;
  actionLabel: string;
  onAction: () => void;
  secondaryLabel?: string;
  onSecondary?: () => void;
  icon?: keyof typeof Feather.glyphMap;
  /** Icon shown inside the primary button. */
  actionIcon?: keyof typeof Feather.glyphMap;
};

/**
 * Calm full-screen message — used for "I didn't catch that", connection
 * trouble, and microphone permission. Same warm styling in every case, so a
 * hiccup never feels like a failure.
 */
export function MessageView({
  heading,
  support,
  actionLabel,
  onAction,
  secondaryLabel,
  onSecondary,
  icon = 'refresh-cw',
  actionIcon = 'mic',
}: MessageViewProps) {
  return (
    <View style={styles.container}>
      <View style={styles.body}>
        <View style={styles.iconWrap}>
          <Feather name={icon} size={30} color={colors.primary} />
        </View>
        <Text style={styles.heading} accessibilityRole="header">
          {heading}
        </Text>
        <Text style={styles.support}>{support}</Text>
      </View>

      <View style={styles.actions}>
        <Button
          label={actionLabel}
          onPress={onAction}
          icon={<Feather name={actionIcon} size={20} color={colors.textOnPrimary} />}
        />
        {secondaryLabel && onSecondary ? (
          <Button label={secondaryLabel} variant="quiet" size="medium" onPress={onSecondary} />
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  body: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  iconWrap: {
    width: 72,
    height: 72,
    borderRadius: radius.md,
    backgroundColor: colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  heading: {
    ...type.title,
    color: colors.text,
    textAlign: 'center',
  },
  support: {
    ...type.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing.sm,
    maxWidth: 320,
  },
  actions: { gap: spacing.sm },
});
