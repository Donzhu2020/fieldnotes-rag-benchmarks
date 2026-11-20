const $ = (id) => document.getElementById(id);
const form = $('search-form');
const results = $('results');
let ready = false;
let busy = false;
function node(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function notice(message) {
  $('notice').textContent = message;
  $('notice').hidden = !message;
}
async function status() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Could not read the local server status.');
    const data = await response.json();
    ready = data.ready;
    $('count').textContent = data.count.toLocaleString('en-US');
    $('sidebar-count').textContent = data.count.toLocaleString('en-US') + ' passages';
    $('memory').textContent = (data.memory_bytes / 1024 / 1024).toFixed(2) + ' MiB';
    $('dimensions').textContent = data.dimensions.toLocaleString('en-US');
    $('vector-format').textContent = `${data.dtype === 'float16' ? 'Float16' : 'Float32'} · Fully loaded`;
    $('corpus-scope').textContent = `${data.count.toLocaleString('en-US')} searchable passages / 1M passages on disk`;
    if (!ready) notice('Vectors are not ready. Run python3 prepare.py in the project directory, then restart the app.');
  } catch (error) { notice(error.message); }
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = $('query').value.trim();
  if (!query || busy) return;
  if (!ready) { notice('No vectors are available yet. Please prepare the corpus first.'); return; }
  busy = true;
  $('submit').disabled = true;
  $('submit').textContent = 'Searching…';
  $('output').setAttribute('aria-busy', 'true');
  notice('');
  results.replaceChildren();
  const loading = node('div', undefined, 'loading');
  loading.append(node('span', '', 'loading-dot'), node('span', 'Embedding your query and scanning every passage…'));
  results.append(loading);
  try {
    const response = await fetch('/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query, k: Number($('result-count').value)}), signal: AbortSignal.timeout(110000)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Search failed. Please try again shortly.');
    results.replaceChildren();
    $('result-title').textContent = data.results.length ? data.results.length + ' ranked passages' : 'No passages available';
    const summary = node('div', undefined, 'result-summary');
    for (const text of [`${data.scanned.toLocaleString('en-US')} passages scanned`, `Full scan ${data.scan_ms.toFixed(2)} ms`, `Query embedding ${data.embedding_ms.toFixed(0)} ms${data.cached ? ' · cached' : ''}`, `Total ${(data.total_ms / 1000).toFixed(2)} s`]) summary.append(node('span', text));
    results.append(summary);
    if (!data.results.length) {
      const empty = node('div', undefined, 'empty');
      empty.append(node('h3', 'No passages available'), node('p', 'The current corpus does not contain searchable passages.'));
      results.append(empty);
    }
    data.results.forEach((item, index) => {
      const card = node('article', undefined, 'result-card');
      card.append(node('span', String(index + 1).padStart(2, '0'), 'result-rank'));
      const content = node('div');
      const meta = node('div', undefined, 'result-meta');
      meta.append(node('span', 'MS MARCO / PASSAGE ' + item.id), node('span', 'Cosine similarity ' + item.score.toFixed(4), 'score'));
      const actions = node('div', undefined, 'result-actions');
      actions.append(node('span', 'Original English passage · Not generated or rewritten'));
      const copy = node('button', 'Copy passage', 'copy-button');
      copy.type = 'button';
      copy.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(item.text); copy.textContent = 'Copied'; }
        catch { copy.textContent = 'Could not copy. Please select the text.'; }
      });
      actions.append(copy);
      content.append(meta, node('p', item.text, 'result-text'), actions);
      card.append(content);
      results.append(card);
    });
    notice(`Searching the first ${data.scanned.toLocaleString('en-US')} passages. Results are ranked by cosine similarity; a higher score is not a guarantee that a passage answers your question.`);
  } catch (error) {
    results.replaceChildren();
    $('result-title').textContent = 'Search could not be completed';
    notice(error.name === 'TimeoutError' ? 'The service timed out. Retrying may incur an additional charge.' : error.message);
  } finally {
    busy = false;
    $('submit').disabled = false;
    $('submit').replaceChildren(document.createTextNode('Search '), node('span', '↗'));
    $('output').setAttribute('aria-busy', 'false');
  }
});
document.querySelectorAll('[data-query]').forEach((button) => button.addEventListener('click', () => {
  if (busy) return;
  $('query').value = button.dataset.query;
  $('query').focus();
  form.requestSubmit();
}));
$('query').addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') form.requestSubmit();
});
status();
