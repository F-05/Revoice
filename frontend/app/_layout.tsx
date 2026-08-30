import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
  useFonts,
} from '@expo-google-fonts/inter';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { colors } from '../constants/theme';
import { hydrateSettings } from '../services/settings';

SplashScreen.preventAutoHideAsync().catch(() => undefined);

export default function RootLayout() {
  const [fontsLoaded, fontError] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
  });

  /**
   * Preferences are read before anything renders. Nothing can start a
   * recording — and therefore nothing can reach the auto-speak decision —
   * until the stored value is in hand, so that decision never runs against a
   * default that is about to be replaced.
   */
  const [settingsReady, setSettingsReady] = useState(false);
  useEffect(() => {
    hydrateSettings().finally(() => setSettingsReady(true));
  }, []);

  const ready = (fontsLoaded || Boolean(fontError)) && settingsReady;

  useEffect(() => {
    if (ready) {
      SplashScreen.hideAsync().catch(() => undefined);
    }
  }, [ready]);

  if (!ready) {
    return <View style={{ flex: 1, backgroundColor: colors.background }} />;
  }

  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.background },
          animation: 'fade',
          animationDuration: 220,
        }}
      />
    </SafeAreaProvider>
  );
}
