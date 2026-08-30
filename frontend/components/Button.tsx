import { ReactNode } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from 'react-native';
import { MIN_TOUCH_TARGET, colors, radius, shadow, spacing, type } from '../constants/theme';

type ButtonProps = {
  label: string;
  onPress: () => void;
  /** `primary` is the one obvious action on a screen. */
  variant?: 'primary' | 'secondary' | 'quiet';
  icon?: ReactNode;
  disabled?: boolean;
  loading?: boolean;
  accessibilityHint?: string;
  style?: ViewStyle;
  /** Full-width, extra tall — used for the main call to action. */
  size?: 'large' | 'medium';
};

export function Button({
  label,
  onPress,
  variant = 'primary',
  icon,
  disabled = false,
  loading = false,
  accessibilityHint,
  style,
  size = 'large',
}: ButtonProps) {
  const isPrimary = variant === 'primary';

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: disabled || loading, busy: loading }}
      style={({ pressed }) => [
        styles.base,
        size === 'large' ? styles.large : styles.medium,
        variant === 'primary' && styles.primary,
        variant === 'primary' && shadow.button,
        variant === 'secondary' && styles.secondary,
        variant === 'quiet' && styles.quiet,
        pressed && (isPrimary ? styles.primaryPressed : styles.secondaryPressed),
        (disabled || loading) && styles.disabled,
        style,
      ]}>
      <View style={styles.content}>
        {loading ? (
          <ActivityIndicator color={isPrimary ? colors.textOnPrimary : colors.primary} />
        ) : (
          <>
            {icon ? <View style={styles.icon}>{icon}</View> : null}
            <Text
              style={[
                type.button,
                { color: isPrimary ? colors.textOnPrimary : colors.text },
                variant === 'quiet' && { color: colors.textSecondary },
              ]}
              numberOfLines={2}>
              {label}
            </Text>
          </>
        )}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    borderRadius: radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.lg,
  },
  large: { paddingVertical: spacing.md + 2, minHeight: 60 },
  medium: { paddingVertical: spacing.md, minHeight: MIN_TOUCH_TARGET },
  primary: { backgroundColor: colors.primary },
  primaryPressed: { backgroundColor: colors.primaryPressed },
  secondary: {
    backgroundColor: colors.surface,
    borderWidth: 1.5,
    borderColor: colors.border,
  },
  secondaryPressed: { backgroundColor: colors.surfaceMuted },
  quiet: { backgroundColor: 'transparent' },
  disabled: { opacity: 0.5 },
  content: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  icon: { marginRight: spacing.sm },
});
