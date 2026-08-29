/**
 * The only place that talks to the backend.
 *
 * Components never call fetch directly — they call these functions and get
 * back plain typed data.
 */
import { Platform } from 'react-native';
import { API_BASE_URL, DEMO_MODE, REQUEST_TIMEOUT_MS, devLog, devWarn } from './config';
import { MOCK_RESPONSES, nextDemoScenario } from './mockResponses';
import type { ProcessSpeechResponse, RepairCandidate, RepairDecision } from '../types/speech';

export class ApiError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Turn a possibly-relative `audio_url` into something playable. */
export function resolveAudioUrl(audioUrl: string | null | undefined): string | null {
  if (!audioUrl) return null;
  if (/^https?:\/\//i.test(audioUrl) || audioUrl.startsWith('file:')) return audioUrl;
  if (audioUrl.startsWith('blob:') || audioUrl.startsWith('data:')) return audioUrl;
  return `${API_BASE_URL}${audioUrl.startsWith('/') ? '' : '/'}${audioUrl}`;
}

function guessUpload(uri: string) {
  const extension = (uri.split('.').pop() ?? 'm4a').split('?')[0].toLowerCase();
  const mimeTypes: Record<string, string> = {
    m4a: 'audio/m4a',
    mp4: 'audio/mp4',
    caf: 'audio/x-caf',
    wav: 'audio/wav',
    webm: 'audio/webm',
    '3gp': 'audio/3gpp',
    aac: 'audio/aac',
    ogg: 'audio/ogg',
  };
  const known = mimeTypes[extension];
  return {
    name: `recording.${known ? extension : 'm4a'}`,
    type: known ?? 'audio/m4a',
  };
}

/** Extension implied by a MIME type, for naming a web blob the backend accepts. */
function extensionForMime(mime: string): string {
  const [type] = mime.split(';');
  const map: Record<string, string> = {
    'audio/webm': 'webm',
    'video/webm': 'webm',
    'audio/ogg': 'ogg',
    'audio/mp4': 'mp4',
    'audio/m4a': 'm4a',
    'audio/x-m4a': 'm4a',
    'audio/aac': 'aac',
    'audio/wav': 'wav',
    'audio/x-wav': 'wav',
    'audio/wave': 'wav',
    'audio/mpeg': 'mp3',
  };
  return map[type.trim().toLowerCase()] ?? 'webm';
}

/**
 * Build the multipart body for one recording.
 *
 * Native and web take different paths on purpose: React Native uploads a file
 * by `{ uri, name, type }`, while on web the recorder hands back a blob: URL
 * that has to be fetched and appended as a real Blob.
 */
async function buildFormData(audioUri: string): Promise<FormData> {
  const form = new FormData();

  if (Platform.OS === 'web') {
    const blob = await fetch(audioUri).then((response) => response.blob());
    const name = `recording.${extensionForMime(blob.type || 'audio/webm')}`;
    devLog(`upload: web blob ${name} type=${blob.type || 'unknown'} bytes=${blob.size}`);
    // Field name is `audio`, per the backend contract.
    form.append('audio', blob, name);
    return form;
  }

  const upload = guessUpload(audioUri);
  devLog(`upload: native file ${upload.name} type=${upload.type}`);
  form.append('audio', {
    uri: audioUri,
    name: upload.name,
    type: upload.type,
  } as unknown as Blob);
  return form;
}

type RawResponse = { ok: boolean; status: number; text: string };

/**
 * POST the multipart body and hand back the raw result.
 *
 * Web uses `fetch`. Native deliberately does not: Expo SDK 57 replaces the
 * global `fetch` with its WinterCG implementation, which cannot serialise
 * React Native's `{ uri, name, type }` file part and throws "Unsupported
 * FormDataPart implementation". XMLHttpRequest goes through React Native's own
 * networking layer, which understands that part and streams the file from disk
 * instead of loading it into JS memory.
 */
function postForm(endpoint: string, form: FormData): Promise<RawResponse> {
  if (Platform.OS === 'web') {
    return postFormWithFetch(endpoint, form);
  }
  return postFormWithXhr(endpoint, form);
}

async function postFormWithFetch(endpoint: string, form: FormData): Promise<RawResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: form,
      // Content-Type is deliberately unset: the runtime adds the multipart
      // boundary, and overriding it breaks the upload.
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    return { ok: response.ok, status: response.status, text: await response.text() };
  } finally {
    clearTimeout(timeout);
  }
}

function postFormWithXhr(endpoint: string, form: FormData): Promise<RawResponse> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', endpoint);
    request.timeout = REQUEST_TIMEOUT_MS;
    request.setRequestHeader('Accept', 'application/json');
    request.onload = () =>
      resolve({
        ok: request.status >= 200 && request.status < 300,
        status: request.status,
        text: request.responseText ?? '',
      });
    request.onerror = () => reject(new ApiError('Could not reach Revoice.'));
    request.ontimeout = () => reject(new ApiError('The request took too long.'));
    request.send(form as unknown as Document);
  });
}

/**
 * Backend error codes that mean "that recording was not usable" rather than
 * "something is broken". They deserve the retry screen, not an error screen.
 */
const UNUSABLE_AUDIO_CODES = ['invalid_audio', 'audio_decode_failed', 'audio_too_large'];

type ErrorEnvelope = { error?: { code?: string; message?: string } };

/** Pull the backend's error code and message out of its documented envelope. */
function readError(status: number, body: string): { code: string; description: string } {
  try {
    const parsed = JSON.parse(body) as ErrorEnvelope;
    if (parsed?.error?.message) {
      const code = parsed.error.code ?? 'error';
      return { code, description: `${code}: ${parsed.error.message}` };
    }
  } catch {
    // Not JSON — fall through to the status line.
  }
  return { code: `http_${status}`, description: `HTTP ${status}` };
}

/** A `retry` response the frontend synthesises when the upload was unusable. */
const RETRY_RESPONSE: ProcessSpeechResponse = {
  status: 'retry',
  raw_transcript: null,
  repaired_text: null,
  confidence: null,
  alternatives: [],
  decision: 'retry',
  repair_available: null,
  uncertain_words: [],
  audio_url: null,
};

const DECISIONS: RepairDecision[] = ['high', 'medium', 'low', 'retry'];

/**
 * Keep only well-formed candidates.
 *
 * A candidate with no text is dropped; a candidate with a non-numeric score
 * keeps its text and reports `null` rather than being given a made-up number.
 */
function readAlternatives(raw: unknown): RepairCandidate[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    const text = typeof item?.text === 'string' ? item.text.trim() : '';
    if (!text) return [];
    return [{
      text,
      confidence: typeof item?.confidence === 'number' ? item.confidence : null,
    }];
  });
}

/** Coerce whatever the backend sent into the shape the app relies on. */
function normalize(raw: Partial<ProcessSpeechResponse> | null): ProcessSpeechResponse {
  const status = raw?.status;
  return {
    status: status === 'success' || status === 'uncertain' || status === 'retry' ? status : 'retry',
    raw_transcript: raw?.raw_transcript ?? null,
    repaired_text: raw?.repaired_text ?? null,
    confidence: typeof raw?.confidence === 'number' ? raw.confidence : null,
    alternatives: readAlternatives(raw?.alternatives),
    // Absent on a backend that has not shipped the field yet — the app derives
    // a band from `status` and `confidence` in that case.
    decision: DECISIONS.includes(raw?.decision as RepairDecision)
      ? (raw!.decision as RepairDecision)
      : null,
    repair_available: typeof raw?.repair_available === 'boolean' ? raw.repair_available : null,
    uncertain_words: Array.isArray(raw?.uncertain_words) ? raw!.uncertain_words! : [],
    audio_url: raw?.audio_url ?? null,
  };
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Send a recording to the backend and get the clarified result.
 *
 * @param audioUri local file URI produced by the recorder.
 */
export async function processSpeech(audioUri: string): Promise<ProcessSpeechResponse> {
  if (DEMO_MODE) {
    devLog('processSpeech: DEMO_MODE is on — returning a mocked response.');
    // Long enough for the processing screen to read as real work, short
    // enough that a live demo never drags.
    await delay(1900);
    return MOCK_RESPONSES[nextDemoScenario()];
  }

  const endpoint = `${API_BASE_URL}/process-speech`;
  const started = Date.now();

  try {
    devLog(`POST ${endpoint} <- ${audioUri}`);
    const form = await buildFormData(audioUri);
    const response = await postForm(endpoint, form);

    if (!response.ok) {
      const { code, description } = readError(response.status, response.text);
      devWarn(`POST /process-speech failed — ${description}`);
      if (UNUSABLE_AUDIO_CODES.includes(code)) {
        // Nothing wrong with the connection — the recording was just no good.
        return RETRY_RESPONSE;
      }
      throw new ApiError(`Server responded with ${description}`);
    }

    const result = normalize(JSON.parse(response.text) as Partial<ProcessSpeechResponse>);
    devLog(
      `response: status=${result.status} confidence=${result.confidence ?? 'n/a'} ` +
        `audio_url=${result.audio_url ?? 'none'} in ${Date.now() - started}ms`,
      result.raw_transcript,
    );
    return result;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if ((error as Error)?.name === 'AbortError') {
      devWarn(`request timed out after ${REQUEST_TIMEOUT_MS}ms`);
      throw new ApiError('The request took too long.', error);
    }
    devWarn(`could not reach ${endpoint}`, error);
    throw new ApiError('Could not reach Revoice.', error);
  }
}
