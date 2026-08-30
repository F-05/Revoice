import { useRouter } from 'expo-router';
import { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';
import { BrandMark } from '../components/BrandMark';
import { Button } from '../components/Button';
import { Screen } from '../components/Screen';
import { colors, spacing, type } from '../constants/theme';
import { markOnboardingSeen } from '../services/history';

export default function Onboarding() {
  const router = useRouter();
  const appear = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(appear, {
      toValue: 1,
      duration: 520,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [appear]);

  const handleGetStarted = () => {
    markOnboardingSeen();
    router.replace('/home');
  };

  return (
    <Screen bottomPadding={spacing.xs}>
      <View style={styles.spacerTop} />

      <Animated.View
        style={[
          styles.hero,
          {
            opacity: appear,
            transform: [
              { translateY: appear.interpolate({ inputRange: [0, 1], outputRange: [16, 0] }) },
            ],
          },
        ]}>
        <BrandMark size={71} />
        <Text style={styles.title} accessibilityRole="header">
          Revoice
        </Text>
        <Text style={styles.tagline}>Your voice, made clear.</Text>
        <Text style={styles.paragraph}>
          Revoice helps make difficult-to-understand speech clearer while preserving what you
          meant.
        </Text>
      </Animated.View>

      <View style={styles.spacerBottom} />

      <Button
        label="Get started"
        onPress={handleGetStarted}
        accessibilityHint="Opens Revoice and takes you to the microphone"
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  spacerTop: { flex: 1.05 },
  spacerBottom: { flex: 1.6 },
  hero: { alignItems: 'center' },
  title: {
    ...type.display,
    color: colors.text,
    marginTop: spacing.lg,
  },
  tagline: {
    ...type.tagline,
    color: colors.primary,
    marginTop: spacing.sm,
  },
  paragraph: {
    ...type.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing.lg,
    maxWidth: 306,
  },
});
