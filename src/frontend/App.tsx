import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import Markdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { askQuestion, connecting, fetchIndexStatus } from './api';
import type { AnswerMode, ChatMessage, ChatResponse, IndexStatus, Source } from './api';
import PdfPreview from './PdfPreview';
import Sources from './Sources';
import remarkCitations from './remarkCitations';

const TITLE = 'RCFNS Sentinel';
const THEME_KEY = 'rcfns-theme';
const MAX_QUESTION_LENGTH = 4000;
// Give the server's 240-second deadline time to return its error first.
const CHAT_TIMEOUT_MS = 250_000;

interface TurnDetails {
  id: string;
  question: string;
  mode: AnswerMode;
}

type Turn = TurnDetails & (
  | { kind: 'pending' }
  | { kind: 'stopped' }
  | { kind: 'answered'; result: ChatResponse }
  | { kind: 'error'; message: string }
);

interface Preview {
  number: number;
  source: Source & { url: string };
}

/** Refresh quickly during import without overlapping requests. */
function useIndexStatus() {
  const [index, setIndex] = useState(connecting);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function refresh() {
      let delay = 2500;
      try {
        const data = await fetchIndexStatus(controller.signal);
        if (controller.signal.aborted) return;
        setIndex(data);
        if (data.complete) delay = 15000;
      } catch {
        if (!controller.signal.aborted) setIndex(connecting);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, delay);
      }
    }

    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  return index;
}

function LibraryStatus({ index }: { index: IndexStatus }) {
  const progress = Math.min(index.indexed + index.failed, index.total);
  const displayCount = index.indexed.toLocaleString();
  return (
    <details className="library">
      <summary className="index-badge" data-ready={index.complete && index.ready}>
        <span className="status-dot" aria-hidden="true" />
        {index.complete || index.ready ? displayCount + ' indexed' : index.total ? 'Indexing' : 'Library status'}
        <span className="chevron" aria-hidden="true" />
      </summary>
      <div className="library-panel">
        <strong>Document library</strong>
        <p role="status">{index.message}</p>
        {index.total > 0 && (
          <>
            {!index.complete && <progress aria-label="Index progress" value={progress} max={index.total} />}
            <dl className="library-counts">
              <div><dt>Indexed</dt><dd>{displayCount} of {index.total.toLocaleString()}</dd></div>
              {!index.complete && <div><dt>Uploaded</dt><dd>{index.uploaded.toLocaleString()}</dd></div>}
              {index.failed > 0 && <div><dt>Failed</dt><dd>{index.failed.toLocaleString()}</dd></div>}
              {index.skipped > 0 && <div><dt>Needs recovery</dt><dd>{index.skipped.toLocaleString()}</dd></div>}
              {index.inventory > 0 && <div><dt><a href="/inventory" target="_blank" rel="noreferrer">File inventory</a></dt><dd>{index.inventory.toLocaleString()}</dd></div>}
            </dl>
          </>
        )}
      </div>
    </details>
  );
}

const table: Components['table'] = ({ node: _node, ...props }) => (
  <div className="table-scroll"><table {...props} /></div>
);

function Answer({ result, requestedMode }: { result: ChatResponse; requestedMode: AnswerMode }) {
  const [preview, setPreview] = useState<Preview | null>(null);
  const previewPanel = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const panel = previewPanel.current;
    if (!preview || !panel) return;
    panel.open = true;
    panel.scrollIntoView({ block: 'nearest' });
  }, [preview]);

  const components = useMemo<Components>(() => ({
    table,
    a: ({ node, href, children, ...props }) => {
      const number = node?.properties.dataCitation;
      const source = typeof number === 'number' ? result.sources[number - 1] : undefined;
      const url = source?.url;
      if (!source || !url || typeof number !== 'number') return <a href={href} {...props}>{children}</a>;

      return (
        <a
          className="citation-link"
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={'Citation ' + number + ', ' + source.name}
          onClick={event => {
            if (!source.regions.length || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
            event.preventDefault();
            setPreview({ source: { ...source, url }, number });
          }}
        >
          {children}
        </a>
      );
    },
  }), [result.sources]);

  return (
    <>
      {requestedMode === 'agentic' && result.mode === 'standard' && result.sources.length > 0 && (
        <p className="mode-note">Used Standard search.</p>
      )}
      <div className="answer">
        <Markdown
          remarkPlugins={[remarkGfm, [remarkCitations, { sources: result.sources }]]}
          skipHtml
          disallowedElements={['img']}
          components={components}
        >
          {result.answer}
        </Markdown>
      </div>
      {preview && <PdfPreview key={preview.number} source={preview.source} panelRef={previewPanel} />}
      <Sources sources={result.sources} />
    </>
  );
}

const ChatTurn = memo(function ChatTurn({ turn }: { turn: Turn }) {
  return (
    <div className="turn">
      <article className="message message-user" aria-label="Your question">
        <span className="speaker">You</span>
        <p>{turn.question}</p>
      </article>
      <article className="message message-assistant" aria-label="Sentinel response">
        <div className="message-header">
          <span className="speaker">Sentinel</span>
          {turn.kind === 'answered' && turn.result.mode === 'agentic' && (
            <span className="answer-mode">Agentic</span>
          )}
        </div>
        {turn.kind === 'pending' && (
          <p className="message-state" role="status">
            {turn.mode === 'agentic' ? 'Investigating the indexed files...' : 'Searching the indexed files...'}
          </p>
        )}
        {turn.kind === 'stopped' && <p className="message-state" role="status">Stopped waiting for an answer.</p>}
        {turn.kind === 'error' && <p className="message-error" role="alert">{turn.message}</p>}
        {turn.kind === 'answered' && <Answer result={turn.result} requestedMode={turn.mode} />}
      </article>
    </div>
  );
});

/** Collect at most three completed exchanges, regardless of the thread length. */
function recentHistory(turns: Turn[]): ChatMessage[] {
  const history: ChatMessage[] = [];
  for (let index = turns.length - 1; index >= 0 && history.length < 6; index--) {
    const turn = turns[index];
    if (turn?.kind !== 'answered') continue;
    history.unshift(
      { role: 'user', content: turn.question },
      { role: 'assistant', content: turn.result.answer.slice(0, 8000) },
    );
  }
  return history;
}

export default function App() {
  const index = useIndexStatus();
  const [theme, setTheme] = useState<'light' | 'dark'>(() => localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState('');
  const [mode, setMode] = useState<AnswerMode>('standard');
  const newestTurn = useRef<HTMLDivElement>(null);
  const activeRequest = useRef<AbortController | null>(null);
  const pending = turns.at(-1)?.kind === 'pending';
  const turnCount = turns.length;
  const canSend = index.ready && !pending && !!draft.trim();

  useEffect(() => {
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (turnCount) newestTurn.current?.scrollIntoView({ block: 'start' });
  }, [turnCount]);

  useEffect(() => () => activeRequest.current?.abort(), []);

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSend || activeRequest.current) return;
    const question = draft.trim();
    const turn: TurnDetails = { id: crypto.randomUUID(), question, mode };
    const request = new AbortController();
    activeRequest.current = request;
    setDraft('');
    setTurns(items => [...items, { ...turn, kind: 'pending' }]);

    const timeout = new Error('The search took too long. Try again.');
    const timer = setTimeout(() => request.abort(timeout), CHAT_TIMEOUT_MS);
    let finished: Turn;
    try {
      const result = await askQuestion(question, mode, recentHistory(turns), request.signal);
      finished = { ...turn, kind: 'answered', result };
    } catch (error) {
      const reason: unknown = request.signal.reason;
      if (request.signal.aborted && reason !== timeout) {
        finished = { ...turn, kind: 'stopped' };
      } else {
        const message = reason === timeout
          ? timeout.message
          : error instanceof Error ? error.message : 'The question could not be answered.';
        finished = { ...turn, kind: 'error', message };
      }
      setDraft(current => current || question);
    } finally {
      clearTimeout(timer);
      activeRequest.current = null;
    }
    setTurns(items => items.map(item => item.id === turn.id ? finished : item));
  }

  function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <div className="app" data-theme={theme}>
      <header className="title-bar">
        <div className="bar-inner title-inner">
          <span className="brand-mark" aria-hidden="true"><span className="chat-glyph" /></span>
          <h1>{TITLE}</h1>
          <LibraryStatus index={index} />
          <button
            className="theme-toggle"
            type="button"
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <main className="conversation" aria-label="Conversation" tabIndex={0}>
        <div className="thread">
          {turns.length === 0 ? (
            <div className="empty-state">
              <span className="empty-mark" aria-hidden="true"><span className="chat-glyph" /></span>
              <h2>Ask your document library</h2>
              <p role="status">
                {index.ready
                  ? index.complete
                    ? 'Ask a question and follow the citations to the original files.'
                    : 'You can ask about the files indexed so far. More are on the way.'
                  : index.message}
              </p>
            </div>
          ) : (
            <div className="turns" role="log" aria-label="Chat messages">
              {turns.map((turn, position) => (
                <div key={turn.id} ref={position === turnCount - 1 ? newestTurn : null}>
                  <ChatTurn turn={turn} />
                </div>
              ))}
            </div>
          )}
        </div>
      </main>

      <footer className="composer-bar">
        <div className="bar-inner">
          <form className="composer" onSubmit={submitQuestion}>
            <textarea
              aria-label="Question"
              enterKeyHint="send"
              placeholder="Ask about the indexed files..."
              maxLength={MAX_QUESTION_LENGTH}
              rows={1}
              value={draft}
              onChange={event => setDraft(event.target.value)}
              onKeyDown={handleComposerKey}
            />
            <div className="composer-controls">
              <select
                className="mode-select"
                aria-label="Answer mode"
                value={mode}
                onChange={event => setMode(event.target.value === 'agentic' ? 'agentic' : 'standard')}
              >
                <option value="standard">Standard</option>
                <option value="agentic">Agentic</option>
              </select>
              {pending ? (
                <button className="send-button" type="button" onClick={() => activeRequest.current?.abort()}>
                  Stop
                </button>
              ) : (
                <button className="send-button" type="submit" disabled={!canSend}>Send</button>
              )}
            </div>
          </form>
        </div>
      </footer>
    </div>
  );
}
