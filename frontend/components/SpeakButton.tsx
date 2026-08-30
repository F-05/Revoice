import { Feather } from '@expo/vector-icons';
import { Button } from './Button';
import { colors } from '../constants/theme';

type SpeakButtonProps = {
  onPress: () => void;
  isSpeaking: boolean;
  disabled?: boolean;
};

/** Primary action on the success screen: say the sentence out loud. */
export function SpeakButton({ onPress, isSpeaking, disabled }: SpeakButtonProps) {
  return (
    <Button
      label={isSpeaking ? 'Speaking…' : 'Speak'}
      onPress={onPress}
      disabled={disabled}
      accessibilityHint="Plays the clear version of your sentence out loud"
      icon={
        <Feather
          name={isSpeaking ? 'volume-2' : 'play'}
          size={22}
          color={colors.textOnPrimary}
        />
      }
    />
  );
}
