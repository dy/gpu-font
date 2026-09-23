# Glyph lookup probe

Does the text-style encoder already recognise icons? `scripts/glyph-probe.mjs` renders twenty
shared concepts (home, search, settings, …) from six openly licensed icon sets — Material Symbols
Outlined, Rounded and Sharp, Font Awesome Free, Bootstrap Icons, Phosphor — at 64 px as references
and 24 px (UI size) as queries, and ranks every query against all 120 references. Results are in
`bench/glyph-probe.json`.

| Ranking by | Exact icon, top-1 | Exact icon, top-5 | Right set, top-1 |
| --- | --- | --- | --- |
| Deployed encoder | 75.0% | 100% | 75.0% |
| Raw 32×32 pixels | 70.8% | 100% | 70.8% |

23 of the encoder's 30 misses confuse Material's Outlined, Rounded and Sharp styles, which differ
only in corners and terminals and blur together at UI size. The encoder is barely better than
pixels and does not group icons by set, which is the answer a designer wants ("Material Symbols
Rounded, home").

This probe is easy on purpose: 120 references and the same icon at another size. It is a floor, not
an accuracy claim. Icon mode needs full-set glyph catalogues (thousands of entries per set), training
with the set and style as the label, and real UI screenshots as queries.
