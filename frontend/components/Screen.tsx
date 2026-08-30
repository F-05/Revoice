import { ReactNode } from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { CONTENT_MAX_WIDTH, colors, spacing } from '../constants/theme';

type ScreenProps = {
  children: ReactNode;
  style?: ViewStyle;
  /** Extra breathing room at the bottom, above the home indicator. */
  bottomPadding?: number;
};

/** Cream page with safe-area padding. Every screen sits inside one. */
export function Screen({ children, style, bottomPadding = spacing.lg }: ScreenProps) {
  const insets = useSafeAreaInsets();
  return (
    <View
      style={[
        styles.screen,
        {
          paddingTop: insets.top + spacing.sm,
          paddingBottom: insets.bottom + bottomPadding,
        },
        style,
      ]}>
      <View style={styles.column}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: spacing.lg,
  },
  /** Keeps text and controls off the edges on wider screens. */
  column: {
    flex: 1,
    width: '100%',
    maxWidth: CONTENT_MAX_WIDTH,
    alignSelf: 'center',
  },
});
