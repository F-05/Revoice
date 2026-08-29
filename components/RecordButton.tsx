import { Feather } from '@expo/vector-icons';
import { useEffect, useRef } from 'react';
import { Animated, Easing, Pressable, StyleSheet, View } from 'react-native';
import { colors, shadow } from '../constants/theme';

type RecordButtonProps = {
  isRecording: boolean;
  onPress: () => void;
  disabled?: boolean;
  /** Diameter of the terracotta circle. */
  size?: number;
};

/**
 * A refined terracotta circle inside a pale ring. The circle stays modest —
 * the comfortable touch target comes from `hitSlop`, not from scaling the
 * artwork up. While recording it breathes gently, and the icon becomes a
 * square so the state is never carried by colour alone.
 */
export function RecordButton({
  isRecording,
  onPress,
  disabled = false,
  size = 104,
}: RecordButtonProps) {
  const pulse = useRef(new Animated.Value(0)).current;
  const press = useRef(new Animated.Value(1)).current;

  const ringSize = size * 1.3;

  useEffect(() => {
    if (!isRecording) {
      pulse.stopAnimation();
      Animated.timing(pulse, { toValue: 0, duration: 260, useNativeDriver: true }).start();
      return;
    }

    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          toValue: 1,
          duration: 1600,
          easing: Easing.out(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(pulse, { toValue: 0, duration: 0, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [isRecording, pulse]);

  return (
    <View style={[styles.wrapper, { width: ringSize, height: ringSize }]}>
      {/* Static pale ring. */}
      <View
        style={[
          styles.ring,
          { width: ringSize, height: ringSize, borderRadius: ringSize / 2 },
        ]}
      />

      {/* Expanding halo, only while recording. */}
      <Animated.View
        style={[
          styles.halo,
          { width: size, height: size, borderRadius: size / 2 },
          {
            opacity: pulse.interpolate({ inputRange: [0, 1], outputRange: [0.45, 0] }),
            transform: [
              { scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.5] }) },
            ],
          },
        ]}
      />

      <Animated.View style={{ transform: [{ scale: press }] }}>
        <Pressable
          onPress={onPress}
          disabled={disabled}
          hitSlop={16}
          onPressIn={() =>
            Animated.spring(press, {
              toValue: 0.95,
              useNativeDriver: true,
              speed: 40,
              bounciness: 0,
            }).start()
          }
          onPressOut={() =>
            Animated.spring(press, {
              toValue: 1,
              useNativeDriver: true,
              speed: 30,
              bounciness: 6,
            }).start()
          }
          accessibilityRole="button"
          accessibilityLabel={isRecording ? 'Stop recording' : 'Start recording'}
          accessibilityHint={
            isRecording
              ? 'Stops recording and makes your words clear'
              : 'Records what you say so Revoice can make it clear'
          }
          accessibilityState={{ disabled, selected: isRecording }}
          style={[
            styles.button,
            { width: size, height: size, borderRadius: size / 2 },
            shadow.button,
            disabled && styles.disabled,
          ]}>
          <Feather
            name={isRecording ? 'square' : 'mic'}
            size={isRecording ? size * 0.28 : size * 0.35}
            color={colors.textOnPrimary}
          />
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { alignItems: 'center', justifyContent: 'center' },
  ring: {
    position: 'absolute',
    pointerEvents: 'none',
    backgroundColor: colors.primarySoft,
    opacity: 0.55,
  },
  halo: {
    position: 'absolute',
    pointerEvents: 'none',
    backgroundColor: colors.primarySoft,
  },
  button: {
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabled: { opacity: 0.6 },
});
