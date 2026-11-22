import type { Source } from './api';

interface SourceFile extends Source {
  numbers: number[];
}

export default function Sources({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;

  const files = new Map<string | number, SourceFile>();
  sources.forEach((source, index) => {
    const key = source.url || source.path || index;
    const file = files.get(key);
    if (file) {
      file.numbers.push(index + 1);
    } else {
      files.set(key, { ...source, numbers: [index + 1] });
    }
  });

  return (
    <details className="sources">
      <summary>{files.size} {files.size === 1 ? 'source' : 'sources'}</summary>
      <ul className="source-files">
        {Array.from(files, ([key, file]) => (
          <li key={key}>
            <span className="source-number" aria-label={'Citations ' + file.numbers.join(', ')}>
              {file.numbers.join(', ')}
            </span>
            <div>
              {file.url ? (
                <a className="source-link" href={file.url} target="_blank" rel="noopener noreferrer">
                  {file.name}
                </a>
              ) : file.name}
              {file.path && file.path !== file.name && (
                <div className="source-path">{file.path}</div>
              )}
            </div>
          </li>
        ))}
      </ul>
    </details>
  );
}
