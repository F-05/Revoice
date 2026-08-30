import { Feather } from '@expo/vector-icons';
import { useEffect, useRef, useState } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';
import { colors, spacing, type } from '../constants/theme';

const STEPS = ['Listening', 'Understanding', 'Clarifying'] as const;
const STEP_DELAY_MS = 650;

/**
 * The processing state. Plain-language steps only — the user never sees model
 * names or technical terms, just calm progress.
 */
export function LoadingState() {
  const [completed, setCompleted] = useState(0);

  useEffect(() => {
    const timers = STEPS.slice(0, -1).map((_, index) =>
      setTimeout(() => setCompleted(index + 1), STEP_DELAY_MS * (index + 1)),
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <View
      style={styles.container}
      accessible
      accessibilityLabel={`Making your words clear. ${STEPS[Math.min(completed, STEPS.length - 1)]}.`}
      accessibilityLiveRegion="polite">
      {STEPS.map((step, index) => (
        <Step
          key={step}
          label={step}
          state={index < completed ? 'done' : index === completed ? 'active' : 'waiting'}
        />
      ))}
    </View>
  );
}

function Step({
  label,
  state,
}: {
  label: string;
  state: 'done' | 'active' | 'waiting';
}) {
  const appear = useRef(new Animated.Value(state === 'waiting' ? 0.25 : 1)).current;
  const spin = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(appear, {
      toValue: state === 'waiting' ? 0.25 : 1,
      duration: 320,
      easing: Easing.out(Easing.ease),
      useNativeDriver: true,
    }).start();
  }, [appear, state]);

  useEffect(() => {
    if (state !== 'active') return;
    const loop = Animated.loop(
      Animated.timing(spin, {
        toValue: 1,
        duration: 1100,
        easing: Easing.linear,
        useNativeDriver: true,
      }),
    );
    spin.setValue(0);
    loop.start();
    return () => loop.stop();
  }, [spin, state]);

  return (
    <Animated.View style={[styles.step, { opacity: appear }]}>
      <View style={styles.marker}>
        {state === 'done' ? (
          <Feather name="check" size={22} color={colors.success} />
        ) : state === 'active' ? (
          <Animated.View
            style={{
              transform: [
                {
                  rotate: spin.interpolate({
                    inputRange: [0, 1],
                    outputRange: ['0deg', '360deg'],
                  }),
                },
              ],
            }}>
            <Feather name="loader" size={22} color={colors.primary} />
          </Animated.View>
        ) : (
          <View style={styles.dot} />
        )}
      </View>
      <Text style={[styles.label, state === 'waiting' && { color: colors.textSecondary }]}>
        {state === 'active' ? `${label}…` : label}
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { gap: spacing.lg },
  step: { flexDirection: 'row', alignItems: 'center' },
  marker: { width: 34, alignItems: 'center' },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.border,
  },
  label: {
    ...type.headline,
    color: colors.text,
    marginLeft: spacing.sm,
  },
});
