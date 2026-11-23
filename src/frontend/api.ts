export type AnswerMode = 'standard' | 'agentic';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface CitationRegion {
  page: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface Source {
  name: string;
  path?: string;
  url?: string;
  regions: CitationRegion[];
}

export interface IndexStatus {
  ready: boolean;
  complete: boolean;
  message: string;
  indexed: number;
  total: number;
  uploaded: number;
  failed: number;
  skipped: number;
  inventory: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  mode: AnswerMode;
}

export const connecting: IndexStatus = {
  ready: false,
  complete: false,
  message: 'Connecting to Sentinel...',
  indexed: 0,
  total: 0,
  uploaded: 0,
  failed: 0,
  skipped: 0,
  inventory: 0,
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isCoordinate(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

function isRegion(value: unknown): value is CitationRegion {
  if (!isObject(value)) return false;
  const { page, left, right, top, bottom } = value;
  return isCoordinate(page) && Number.isInteger(page) && page > 0
    && isCoordinate(left) && isCoordinate(right) && left < right
    && isCoordinate(top) && isCoordinate(bottom) && top < bottom;
}

/** Normalize optional metadata while preserving the answer's citation positions. */
function readSource(value: unknown): Source {
  if (!isObject(value) || typeof value.name !== 'string') {
    return { name: 'Source document', regions: [] };
  }

  return {
    name: value.name,
    path: typeof value.path === 'string' ? value.path : undefined,
    url: typeof value.url === 'string' && /^\/sources\/[A-Za-z0-9_-]+$/.test(value.url)
      ? value.url
      : undefined,
    regions: Array.isArray(value.regions) ? value.regions.filter(isRegion) : [],
  };
}

function count(value: unknown): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : 0;
}

export async function fetchIndexStatus(signal: AbortSignal): Promise<IndexStatus> {
  const response = await fetch('/status', { signal, cache: 'no-store' });
  const data: unknown = await response.json();
  if (!response.ok || !isObject(data) || typeof data.status !== 'string') {
    throw new Error('The index status is unavailable.');
  }

  const indexed = count(data.indexed_count);
  return {
    ready: data.ready === true,
    complete: typeof data.complete === 'boolean' ? data.complete : data.ready === true,
    message: data.status,
    indexed,
    total: Math.max(count(data.total_count), indexed),
    uploaded: count(data.uploaded_count),
    failed: count(data.failed_count),
    skipped: count(data.skipped_count),
    inventory: count(data.inventory_count),
  };
}

export async function askQuestion(
  question: string,
  mode: AnswerMode,
  history: ChatMessage[],
  signal: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, mode, history }),
    signal,
  });
  const data: unknown = await response.json().catch(() => null);
  signal.throwIfAborted();
  if (!isObject(data) || !response.ok || typeof data.answer !== 'string') {
    const detail = isObject(data) && data.detail;
    throw new Error(typeof detail === 'string' ? detail : 'The question could not be answered.');
  }

  return {
    answer: data.answer,
    sources: Array.isArray(data.sources) ? data.sources.map(readSource) : [],
    mode: data.mode_used === 'agentic' ? 'agentic' : 'standard',
  };
}
