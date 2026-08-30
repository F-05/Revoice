import { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, View } from 'react-native';
import { colors } from '../constants/theme';

type AudioWaveformProps = {
  /** Live microphone level, 0..1. Falls back to gentle idle motion at 0. */
  level: number;
  active: boolean;
  barCount?: number;
  height?: number;
};

/**
 * Soft terracotta bars that follow the microphone level. When metering is
 * unavailable the bars still drift gently, so the app always looks like it is
 * listening rather than frozen.
 */
export function AudioWaveform({
  level,
  active,
  barCount = 23,
  height = 96,
}: AudioWaveformProps) {
  const bars = useRef(
    Array.from({ length: barCount }, () => new Animated.Value(0.16)),
  ).current;
  const levelRef = useRef(level);
  levelRef.current = level;

  useEffect(() => {
    if (!active) {
      bars.forEach((bar) => {
        Animated.timing(bar, {
          toValue: 0.12,
          duration: 220,
          useNativeDriver: true,
        }).start();
      });
      return;
    }

    let step = 0;
    const interval = setInterval(() => {
      step += 1;
      bars.forEach((bar, index) => {
        // Centre bars react most, edges taper — reads as a voice, not a chart.
        const centreBias = 1 - Math.abs(index - (barCount - 1) / 2) / ((barCount - 1) / 2);
        const wave = 0.5 + 0.5 * Math.sin(step * 0.55 + index * 0.7);
        const energy = Math.max(levelRef.current, 0.18);
        const target = 0.12 + energy * (0.35 + 0.65 * wave) * (0.45 + 0.55 * centreBias);
        Animated.timing(bar, {
          toValue: Math.min(1, target),
          duration: 170,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        }).start();
      });
    }, 150);

    return () => clearInterval(interval);
  }, [active, barCount, bars]);

  return (
    <View
      style={[styles.container, { height }]}
      accessible
      accessibilityRole="progressbar"
      accessibilityLabel="Recording in progress">
      {bars.map((bar, index) => (
        <Animated.View
          key={index}
          style={[
            styles.bar,
            {
              height,
              transform: [{ scaleY: bar }],
            },
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  bar: {
    width: 6,
    borderRadius: 3,
    backgroundColor: colors.primary,
  },
});
