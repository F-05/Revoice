import { Feather } from '@expo/vector-icons';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { MIN_TOUCH_TARGET, colors, radius, spacing, type } from '../constants/theme';

type SuggestionButtonProps = {
  label: string;
  onPress: () => void;
  selected?: boolean;
  /** Quieter styling for escape-hatch options like "Something else". */
  subtle?: boolean;
};

/** A large, easy-to-hit choice for resolving an uncertain word. */
export function SuggestionButton({
  label,
  onPress,
  selected = false,
  subtle = false,
}: SuggestionButtonProps) {
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ selected }}
      style={({ pressed }) => [
        styles.option,
        subtle && styles.subtle,
        selected && styles.selected,
        pressed && styles.pressed,
      ]}>
      <Text
        style={[
          styles.text,
          subtle && { color: colors.textSecondary },
          selected && { color: colors.primary },
        ]}
        numberOfLines={2}>
        {label}
      </Text>
      {selected ? (
        <View style={styles.check}>
          <Feather name="check" size={20} color={colors.primary} />
        </View>
      ) : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 68,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
    borderWidth: 1.5,
    borderColor: colors.border,
  },
  subtle: { backgroundColor: 'transparent' },
  selected: {
    borderColor: colors.primary,
    backgroundColor: colors.primarySoft,
  },
  pressed: { backgroundColor: colors.surfaceMuted },
  text: {
    ...type.button,
    color: colors.text,
    flexShrink: 1,
  },
  check: { marginLeft: spacing.sm },
});
