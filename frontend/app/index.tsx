import { Redirect } from 'expo-router';
import { useEffect, useState } from 'react';
import { View } from 'react-native';
import { colors } from '../constants/theme';
import { hasSeenOnboarding } from '../services/history';

/** Entry point: onboarding the first time, straight to the microphone after. */
export default function Index() {
  const [seen, setSeen] = useState<boolean | null>(null);

  useEffect(() => {
    hasSeenOnboarding().then(setSeen);
  }, []);

  if (seen === null) {
    return <View style={{ flex: 1, backgroundColor: colors.background }} />;
  }

  return <Redirect href={seen ? '/home' : '/onboarding'} />;
}
