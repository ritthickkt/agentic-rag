# Incomplete runs

Files here did not score their whole split, so their metrics are computed over a
smaller denominator than the split contains and are **not comparable** to anything in
`results/`. They are kept only for debugging.

- `product_dev.INCOMPLETE-137-of-420.json` — 2026-09-27. 283 of 420 questions failed
  with HTTP 400 "credit balance is too low". The 137 that completed scored 83.2%
  overall, but that figure describes 137 self-selected questions (whichever ran before
  the balance hit zero), not the dev split.
