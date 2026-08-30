import { useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';
import { AppHeader, HeaderAction } from '../components/AppHeader';
import { Screen } from '../components/Screen';
import { colors, radius, spacing, type } from '../constants/theme';
import { useSettings } from '../hooks/useSettings';

export default function SettingsScreen() {
  const router = useRouter();
  const { settings, updateSettings } = useSettings();

  const setSpeakAutomatically = (value: boolean) => {
    updateSettings({ speakAutomatically: value });
  };

  return (
    <Screen>
      <AppHeader right={<HeaderAction label="Close" onPress={() => router.back()} />} />

      <Text style={styles.title} accessibilityRole="header">
        Settings
      </Text>

      <ScrollView contentContainerStyle={styles.body} showsVerticalScrollIndicator={false}>
        <Text style={styles.section}>Speech</Text>

        {/*
          The whole row toggles, not just the switch. A 40pt switch is a hard
          target for someone with limited motor control, and this app is built
          for exactly that person — so the switch is presentational and the row
          owns the single interaction.
        */}
        <Pressable
          onPress={() => setSpeakAutomatically(!settings.speakAutomatically)}
          accessibilityRole="switch"
          accessibilityLabel="Speak automatically"
          accessibilityHint="Speaks Revoice's best interpretation as soon as it is ready"
          accessibilityState={{ checked: settings.speakAutomatically }}
          style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}>
          <Text style={styles.rowLabel}>Speak automatically</Text>
          <View pointerEvents="none" importantForAccessibility="no-hide-descendants">
            <Switch
              value={settings.speakAutomatically}
              trackColor={{ false: colors.border, true: colors.primary }}
              thumbColor={colors.surface}
              ios_backgroundColor={colors.border}
            />
          </View>
        </Pressable>

        <Text style={styles.footnote}>
          When enabled, Revoice speaks its best interpretation as soon as it is ready. When
          it is less sure, it still asks first.
        </Text>

        <View style={styles.divider} />

        <Text style={styles.section}>About</Text>
        <Text style={styles.aboutName}>Revoice</Text>
        <Text style={styles.aboutTagline}>Your voice, made clear.</Text>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: {
    ...type.title,
    color: colors.text,
    marginTop: spacing.lg,
    marginBottom: spacing.lg,
  },
  body: { paddingBottom: spacing.xl },
  section: {
    ...type.caption,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.md,
    minHeight: 68,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  rowPressed: { backgroundColor: colors.surfaceMuted },
  rowLabel: {
    ...type.button,
    color: colors.text,
    flexShrink: 1,
  },
  footnote: {
    ...type.support,
    color: colors.textSecondary,
    marginTop: spacing.md,
    paddingHorizontal: spacing.xs,
  },
  divider: {
    height: 1,
    backgroundColor: colors.border,
    marginVertical: spacing.xl,
  },
  aboutName: {
    ...type.button,
    color: colors.text,
  },
  aboutTagline: {
    ...type.support,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
});
