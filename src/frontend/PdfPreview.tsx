import { useEffect, useMemo, useRef, useState } from 'react';
import type { Ref } from 'react';
import type { PDFDocumentLoadingTask, PDFDocumentProxy, RenderTask } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import type { CitationRegion, Source } from './api';

interface PageSize {
  width: number;
  height: number;
}

interface PageProps {
  pdf: PDFDocumentProxy;
  number: number;
  regions: CitationRegion[];
}

const MAX_RENDER_EDGE = 1600;

function PdfPage({ pdf, number, regions }: PageProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const firstHighlight = useRef<HTMLElement>(null);
  const [size, setSize] = useState<PageSize | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    let render: RenderTask | undefined;

    /** Bound the canvas size and cancel obsolete page renders during navigation. */
    async function draw() {
      try {
        const page = await pdf.getPage(number);
        const target = canvas.current;
        if (!active || !target) return;
        const original = page.getViewport({ scale: 1 });
        const scale = Math.min(2, MAX_RENDER_EDGE / Math.max(original.width, original.height));
        const viewport = page.getViewport({ scale });
        target.width = Math.ceil(viewport.width);
        target.height = Math.ceil(viewport.height);
        render = page.render({ canvas: target, viewport });
        await render.promise;
        if (active) setSize({ width: original.width, height: original.height });
      } catch {
        if (active) setFailed(true);
      }
    }

    void draw();
    return () => {
      active = false;
      render?.cancel();
    };
  }, [number, pdf]);

  useEffect(() => {
    const highlight = firstHighlight.current;
    const pane = highlight?.closest<HTMLElement>('.pdf-scroll');
    if (!highlight || !pane) return;
    const top = highlight.getBoundingClientRect().top - pane.getBoundingClientRect().top + pane.scrollTop;
    pane.scrollTop = Math.max(0, top - Math.max(12, (pane.clientHeight - highlight.offsetHeight) / 2));
  }, [size]);

  if (failed) return <p role="status">This page could not be previewed. Open the full PDF to view it.</p>;

  return (
    <div className="pdf-page" hidden={!size}>
      <canvas ref={canvas} role="img" aria-label={'Original PDF page ' + number} />
      {size && regions.map(box => {
        const left = Math.min(100, 100 * box.left / size.width);
        const right = Math.min(100, 100 * box.right / size.width);
        const top = Math.min(100, 100 * box.top / size.height);
        const bottom = Math.min(100, 100 * box.bottom / size.height);
        if (right <= left || bottom <= top) return null;
        return (
          <mark
            key={[box.left, box.right, box.top, box.bottom].join(',')}
            ref={box === regions[0] ? firstHighlight : null}
            className="pdf-highlight"
            aria-label="Cited area"
            style={{ left: left + '%', top: top + '%', width: (right - left) + '%', height: (bottom - top) + '%' }}
          />
        );
      })}
    </div>
  );
}

interface PreviewSource extends Source {
  url: string;
}

function PdfViewer({ source }: { source: PreviewSource }) {
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(source.regions[0]?.page ?? 1);
  const [failed, setFailed] = useState(false);
  const regions = useMemo(() => source.regions.filter(region => region.page === page), [source, page]);

  useEffect(() => {
    let active = true;
    let loading: PDFDocumentLoadingTask | undefined;
    async function load() {
      try {
        const pdfjs = await import('pdfjs-dist');
        if (!active) return;
        pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
        loading = pdfjs.getDocument({ url: source.url, enableXfa: false });
        const document = await loading.promise;
        if (active) {
          setPage(current => Math.min(current, document.numPages));
          setPdf(document);
        }
      } catch {
        if (active) setFailed(true);
      }
    }

    void load();
    return () => {
      active = false;
      void loading?.destroy().catch(() => undefined);
    };
  }, [source.url]);

  return (
    <>
      {pdf && (
        <nav className="pdf-navigation" aria-label="PDF pages">
          <button type="button" onClick={() => setPage(page - 1)} disabled={page === 1}>Previous</button>
          <span>Page {page} of {pdf.numPages}</span>
          <button type="button" onClick={() => setPage(page + 1)} disabled={page === pdf.numPages}>Next</button>
        </nav>
      )}
      <div className="pdf-scroll">
        {failed && <p role="status">The preview could not load. Use the full PDF link to open the file.</p>}
        {!pdf && !failed && <p role="status">Loading the original PDF...</p>}
        {pdf && <PdfPage key={page} pdf={pdf} number={page} regions={regions} />}
      </div>
    </>
  );
}

export default function PdfPreview({ source, panelRef }: {
  source: PreviewSource;
  panelRef: Ref<HTMLDetailsElement>;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <details
      className="pdf-preview"
      ref={panelRef}
      onToggle={event => setExpanded(event.currentTarget.open)}
    >
      <summary>PDF preview <span>{source.name}</span></summary>
      {expanded && <PdfViewer source={source} />}
      <div className="pdf-preview-footer">
        <a href={source.url} target="_blank" rel="noopener noreferrer">Open full PDF</a>
      </div>
    </details>
  );
}
