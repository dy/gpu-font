// The Sources table, rendered from bench/foundries.json as committed.
const TERMS = { permitted: 'Allowed', 'none-found': 'No restriction found', 'not-verified': 'Not verified', restricted: 'Restricted', ban: 'Forbidden' }
const WORK = { indexed: 'Indexed', partial: 'Partly indexed', inventoried: 'Inventoried', planned: 'Planned', paused: 'Paused', 'permission-to-request': 'Permission to request', 'awaiting-permission': 'Awaiting permission', declined: 'Declined', excluded: 'Excluded' }
const KINDS = { library: 'Library', foundry: 'Foundry', retailer: 'Retailer', aggregator: 'Aggregator', platform: 'Platform', icons: 'Icon fonts', music: 'Music fonts', math: 'Math fonts' }
const $ = id => document.getElementById(id)
const count = n => n.toLocaleString('en-US')
const plural = (n, one, many) => `${count(n)} ${n === 1 ? one : many}`
function el(tag, { dataset, ...props } = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props)
  Object.assign(node.dataset, dataset)
  node.append(...children.filter(child => child != null && child !== false && child !== ''))
  return node
}
const link = (href, text) => el('a', { href, target: '_blank', rel: 'noopener' }, text)

// A written ban or restriction is always shown with the clause it quotes and where it was read.
function terms({ terms: t }) {
  const cell = el('td', {}, el('span', { className: 'pill', dataset: { terms: t.status } }, TERMS[t.status] ?? t.status))
  if (t.quote || t.note) cell.append(el('details', {}, el('summary', {}, t.quote ? 'Clause' : 'Note'),
    t.quote ? el('blockquote', {}, t.quote) : null, t.note ? el('p', {}, t.note) : null,
    el('p', { className: 'checked' }, t.url ? link(t.url, 'Terms') : null, t.url ? `, read ${t.checked}` : `Checked ${t.checked}`)))
  return cell
}
// Status is how far work has gone, with its family counts; held-locally families are captured, not covered.
function status({ work: w, familiesAvailable }) {
  const families = w.familiesIndexed ?? w.familiesInventoried, label = WORK[w.status] ?? w.status
  const text = families ? `${label}: ${plural(families, 'family', 'families')}` : familiesAvailable ? `${label}; ${count(familiesAvailable)} available` : label
  return el('td', {}, el('span', { className: 'status' }, text),
    w.familiesHeldLocally ? el('span', { className: 'held' }, `${count(w.familiesHeldLocally)} held locally, not shipped`) : null,
    w.reason || w.route ? el('span', { className: 'detail' }, w.reason ?? w.route) : null)
}
function row(source) {
  return el('tr', { id: source.id }, el('th', { scope: 'row' }, link(source.url, source.name)), el('td', {}, source.licence ?? '–'), terms(source), status(source))
}

const data = await (await fetch('./bench/foundries.json')).json()
// Most progress first, then by name.
const order = Object.keys(WORK), rank = source => { const i = order.indexOf(source.work.status); return i < 0 ? order.length : i }
$('sources-rows').replaceChildren(...[...data.sources].sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name)).map(row))
const tally = Object.keys(TERMS).map(status => [status, data.sources.filter(s => s.terms.status === status).length]).filter(([, n]) => n)
const indexed = data.sources.reduce((sum, s) => sum + (s.work.familiesIndexed ?? 0), 0)
$('sources-summary').textContent = `${plural(data.sources.length, 'source', 'sources')} checked, ${plural(indexed, 'family', 'families')} indexed. Terms: ${tally.map(([status, n]) => `${TERMS[status].toLowerCase()} ${count(n)}`).join(', ')}.`
$('sources-updated').textContent = data.updated; $('sources-updated').dateTime = data.updated
