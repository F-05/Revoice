import { Feather } from '@expo/vector-icons';
import { StyleSheet, View } from 'react-native';
import { colors, radius, shadow } from '../constants/theme';

type BrandMarkProps = { size?: number };

/** The rounded terracotta square with a microphone — Revoice's logo. */
export function BrandMark({ size = 71 }: BrandMarkProps) {
  return (
    <View
      accessible
      accessibilityRole="image"
      accessibilityLabel="Revoice"
      style={[
        styles.mark,
        { width: size, height: size, borderRadius: size * 0.29 },
        shadow.soft,
      ]}>
      <Feather name="mic" size={size * 0.48} color={colors.textOnPrimary} />
    </View>
  );
}

const styles = StyleSheet.create({
  mark: {
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
