import type { Parents, Root, RootContent, Text } from 'mdast';
import type { Source } from './api';

/** Convert supported citation markers in Markdown text to links to original files. */
export default function remarkCitations({ sources }: { sources: Source[] }) {
  function linkCitations(text: Text): RootContent[] {
    const parts: RootContent[] = [];
    let offset = 0;
    for (const match of text.value.matchAll(/\[([1-9][0-9]*)\]/g)) {
      const source = sources[Number(match[1]) - 1];
      if (!source?.url) continue;
      if (match.index > offset) {
        parts.push({ type: 'text', value: text.value.slice(offset, match.index) });
      }
      parts.push({
        type: 'link',
        url: source.url,
        data: { hProperties: { dataCitation: Number(match[1]) } },
        children: [{ type: 'text', value: match[0] }],
      });
      offset = match.index + match[0].length;
    }

    if (!parts.length) return [text];
    if (offset < text.value.length) parts.push({ type: 'text', value: text.value.slice(offset) });
    return parts;
  }

  function visit(parent: Parents) {
    if (parent.type === 'link' || parent.type === 'linkReference') return;
    const children: RootContent[] = parent.children;
    for (let index = 0; index < children.length; index++) {
      const child = children[index];
      if (!child) continue;
      if ('children' in child) visit(child);
      if (child.type !== 'text') continue;
      const parts = linkCitations(child);
      if (parts[0] === child) continue;
      children.splice(index, 1, ...parts);
      index += parts.length - 1;
    }
  }

  return (tree: Root) => visit(tree);
}
