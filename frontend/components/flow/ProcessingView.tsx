import { StyleSheet, Text, View } from 'react-native';
import { LoadingState } from '../LoadingState';
import { colors, spacing, type } from '../../constants/theme';

export function ProcessingView() {
  return (
    <View style={styles.container}>
      <Text style={styles.heading} accessibilityRole="header">
        Making your words clear…
      </Text>
      <View style={styles.steps}>
        <LoadingState />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center' },
  heading: {
    ...type.title,
    color: colors.text,
    marginBottom: spacing.xxl,
  },
  steps: { paddingLeft: spacing.xs },
});
