/**
 * Revoice design tokens.
 *
 * Everything visual comes from here so the app stays consistent with the
 * onboarding mockup: warm cream paper, one terracotta accent, black text,
 * soft warm gray for support copy, generous whitespace.
 */

export const colors = {
  /** Warm cream page background. */
  background: '#FAF6F0',
  /** Slightly lifted surface for cards / list rows. */
  surface: '#FFFFFF',
  /** Very soft tint used for calm, non-alarming states. */
  surfaceMuted: '#F3ECE3',
  /** Hairline that reads as a divider without turning into a box. */
  border: '#E7DCCE',

  /** Terracotta — the single accent. Used for the mic, primary buttons, tagline. */
  primary: '#B0522F',
  primaryPressed: '#8F401F',
  /** Terracotta at low opacity, for halos and selected states. */
  primarySoft: '#F0DFD3',

  text: '#1A1A1A',
  textSecondary: '#6E655D',
  textOnPrimary: '#FFFFFF',

  /** Quiet positive marker for completed processing steps. */
  success: '#4F7A52',
} as const;

export const fonts = {
  regular: 'Inter_400Regular',
  medium: 'Inter_500Medium',
  semibold: 'Inter_600SemiBold',
  bold: 'Inter_700Bold',
} as const;

/** 4pt spacing scale. */
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
  xxxl: 64,
} as const;

export const radius = {
  sm: 12,
  md: 20,
  lg: 28,
  pill: 999,
} as const;

/**
 * Type scale, tuned for an iPhone-class screen rather than a browser window.
 * Headings are compact and native-feeling; readability comes from contrast and
 * spacing rather than oversized web-style type.
 */
export const type = {
  /** Onboarding wordmark only. Measured against the Design.pdf mockup. */
  display: { fontFamily: fonts.bold, fontSize: 38, lineHeight: 45, letterSpacing: -0.8 },
  title: { fontFamily: fonts.bold, fontSize: 28, lineHeight: 34, letterSpacing: -0.5 },
  headline: { fontFamily: fonts.semibold, fontSize: 22, lineHeight: 29, letterSpacing: -0.3 },
  /** The clarified sentence — the most important text in the app. */
  speech: { fontFamily: fonts.semibold, fontSize: 27, lineHeight: 36, letterSpacing: -0.4 },
  tagline: { fontFamily: fonts.semibold, fontSize: 18, lineHeight: 25 },
  /** Onboarding paragraph and other running copy. */
  body: { fontFamily: fonts.regular, fontSize: 16, lineHeight: 25 },
  /** Supporting copy inside the app — lighter than `body`, never dominant. */
  support: { fontFamily: fonts.regular, fontSize: 16, lineHeight: 23 },
  button: { fontFamily: fonts.semibold, fontSize: 18, lineHeight: 23 },
  label: { fontFamily: fonts.medium, fontSize: 15, lineHeight: 21 },
  caption: { fontFamily: fonts.medium, fontSize: 13, lineHeight: 18, letterSpacing: 0.1 },
} as const;

/** Minimum comfortable touch target for motor-impaired users. */
export const MIN_TOUCH_TARGET = 56;

/**
 * Text and controls stay within this width and sit centred, so nothing
 * stretches edge to edge on wider screens.
 */
export const CONTENT_MAX_WIDTH = 340;

export const shadow = {
  /** Soft warm lift — never a hard drop shadow. */
  soft: {
    shadowColor: '#5A3A22',
    shadowOpacity: 0.1,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 8 },
    elevation: 4,
  },
  button: {
    shadowColor: '#7A3416',
    shadowOpacity: 0.22,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
    elevation: 6,
  },
} as const;
