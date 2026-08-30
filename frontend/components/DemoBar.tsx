import { Pressable, StyleSheet, Text, View } from 'react-native';
import { colors, radius, spacing, type } from '../constants/theme';
import {
  DEMO_SCENARIOS,
  type DemoScenario,
  getPinnedDemoScenario,
  pinDemoScenario,
} from '../services/mockResponses';
import { useState } from 'react';

/**
 * Only rendered while DEMO_MODE is on. Lets us pin which mocked response the
 * next recording returns, so a live demo can show any state on demand.
 * Nothing else in the UI knows this exists.
 */
export function DemoBar() {
  const [pinned, setPinned] = useState<DemoScenario | null>(getPinnedDemoScenario());

  const cycle = () => {
    setPinned((current) => {
      const order: (DemoScenario | null)[] = [null, ...DEMO_SCENARIOS];
      const next = order[(order.indexOf(current) + 1) % order.length];
      pinDemoScenario(next);
      return next;
    });
  };

  return (
    <View style={styles.row}>
      <Pressable
        onPress={cycle}
        accessibilityRole="button"
        accessibilityLabel={`Demo mode. Next result: ${pinned ?? 'cycling through all states'}. Tap to change.`}
        style={({ pressed }) => [styles.pill, pressed && styles.pressed]}>
        <Text style={styles.text}>Demo · next: {pinned ?? 'cycle'}</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { alignItems: 'center', paddingTop: spacing.sm },
  pill: {
    paddingVertical: spacing.xs + 2,
    paddingHorizontal: spacing.md,
    borderRadius: radius.pill,
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: colors.border,
    opacity: 0.75,
  },
  pressed: { backgroundColor: colors.primarySoft },
  text: { ...type.caption, color: colors.textSecondary },
});
