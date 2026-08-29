import { Feather } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { AppHeader, HeaderAction } from '../components/AppHeader';
import { Screen } from '../components/Screen';
import { colors, radius, spacing, type } from '../constants/theme';
import { loadHistory } from '../services/history';
import { speak, stopSpeaking } from '../services/playback';
import type { HistoryItem } from '../types/speech';

export default function History() {
  const router = useRouter();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  useEffect(() => {
    loadHistory().then(setItems);
    return () => stopSpeaking();
  }, []);

  const handleSpeak = (item: HistoryItem) => {
    setSpeakingId(item.id);
    speak({
      text: item.text,
      audioUrl: item.audioUrl,
      onDone: () => setSpeakingId(null),
      onError: () => setSpeakingId(null),
    }).catch(() => setSpeakingId(null));
  };

  return (
    <Screen>
      <AppHeader
        right={
          <HeaderAction label="Close" onPress={() => router.back()} />
        }
      />

      <Text style={styles.title} accessibilityRole="header">
        Recent
      </Text>

      <FlatList
        data={items}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <Text style={styles.empty}>Sentences you clarify will show up here.</Text>
        }
        renderItem={({ item }) => (
          <Pressable
            onPress={() => handleSpeak(item)}
            accessibilityRole="button"
            accessibilityLabel={`Play: ${item.text}`}
            accessibilityState={{ busy: speakingId === item.id }}
            style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}>
            <Text style={styles.phrase}>{item.text}</Text>
            <View style={styles.play}>
              <Feather
                name={speakingId === item.id ? 'volume-2' : 'play'}
                size={20}
                color={colors.primary}
              />
            </View>
          </Pressable>
        )}
      />
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
  list: { gap: spacing.md, paddingBottom: spacing.xl },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.md,
    minHeight: 76,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  rowPressed: { backgroundColor: colors.surfaceMuted },
  empty: {
    ...type.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing.xl,
  },
  phrase: {
    ...type.button,
    color: colors.text,
    flexShrink: 1,
  },
  play: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
