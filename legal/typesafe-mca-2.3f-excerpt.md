# TypeSafe Master Customer Agreement: benchmark-restriction check

**Finding: the quoted restriction was NOT FOUND in the live MCA (checked 2026-09-20).**

> **Status: resolved 2026-09-21.** The escalation this file asked for happened. Alden's
> decision of 2026-09-21 (`PUBLICATION_PERMISSION_README.md`) is to publish and to send no
> permission request, and the disclosure text in the paper and the article was written
> against this finding rather than against the premise it corrects. The founder decision of
> 2026-09-18 (`founder-decision-2026-09-18.md`) carries a correction header pointing here,
> and the drafted permission email (`typesafe-permission-request-email.md`) is marked never
> sent. The to-do list at the end of this file records what was done and what is still open.

## What the repo previously recorded

Captured 2026-09-18, quoted in spec §3.1, and sourced (per the file it came with)
from `research/2026-09-18-chatgpt-deep-research-jev-use-cases-raw.md`. That research file
and the `research/` folder it sits in are internal working material and are **not part of
the public release**; this file is the released record of what they said and of the check
that followed. The quotation was:

> Customer may not "publish benchmarks or performance information about the Services."

That file carried a to-do for G0: *"fetch the live MCA page ... and confirm the
clause wording above. If the wording has changed, update this file."* This is that
update.

## What the live MCA actually says (verified 2026-09-20)

Source URL: <https://typesafe.ai/legal/mca> (the `rel="canonical"` of the captured page).
Captured 2026-09-20. The sha256 in `typesafe-mca-snapshot-2026-09-20.sha256` is over **our
capture**, not over anything a later fetch will reproduce: the page's version line had
already moved between the 2026-09-18 reading and ours, so a reader confirms this finding by
fetching the agreement and reading clause 2.3(f), not by matching a hash.

- `legal/typesafe-mca-snapshot-2026-09-20.html` (248,788 bytes)
- `legal/typesafe-mca-snapshot-2026-09-20.txt` (41,978 chars)

Reproduce the check against the snapshot. **The two snapshot files are TypeSafe's document
and are not in the public release**, only their sha256 is, so this block runs in the
working repository or against a capture you take yourself; the hash beside it identifies
the bytes we ran it against.

```bash
grep -ci benchmark legal/typesafe-mca-snapshot-2026-09-20.txt   # -> 0
grep -ci publish   legal/typesafe-mca-snapshot-2026-09-20.txt   # -> 0
```

Findings:

| Check | Result |
|---|---|
| Page version line | **Last updated Sep 19, 2026** (the spec recorded 2026-08-27) |
| Occurrences of "benchmark" | **0** |
| Occurrences of "publish" | **0** |
| Occurrences of "performance information" | **0** |
| §2.3 License Restrictions | present, subsections **(a) through (m)** |
| **§2.3(f)** | **"interfere with the operation of the Services"** |

§2.3 prohibits reselling the Services standalone (a), model distillation and
training a competing model (b), reverse engineering (c), derivative works (d),
removing proprietary notices (e), interfering with operation (f), circumventing
access restrictions (g), viruses (h), harming availability (i), exceeding usage
limits (j), undocumented access (k), AUP violations (l), and unlawful use (m).

**No subsection restricts publishing benchmarks or performance information.**

The only publicity-related clause is **§16.4 Publicity**, which prevents either
party using the other's name, brand or logo and prevents announcing that the
parties entered the Agreement, without consent. It says nothing about results.

## What is NOT established

The MCA was updated on **2026-09-19**, which is after our 2026-09-18 capture. So
two explanations remain open and this check cannot separate them:

1. the clause existed and was removed in the 2026-09-19 update; or
2. the clause never existed and entered the spec through the ChatGPT deep-research
   file, which `research/README.md` itself designates as not a source of truth (both
   are internal and withheld from the public release, as noted above).

An attempt to settle this against the Wayback Machine snapshot of 2026-09-16
returned an *"Internet Archive: Temporarily Offline"* page rather than archived
content, so it produced **no evidence either way**. Retry before G8.

## Why this matters

1. **Disclosure accuracy (G8, spec §3.3).** The paper must not state that
   TypeSafe's MCA restricts benchmark publication unless that can be shown from a
   dated snapshot. Asserting a nonexistent clause in a paper that already carries a
   conflict-of-interest disclosure is exactly the kind of error a hostile reviewer
   checks in under a minute.
2. **The founder decision (ADR-001, `founder-decision-2026-09-18.md`).** It accepts
   a documented breach risk and orders all Jev inference to finish before
   publication. If there is no such clause, that premise is void. The decision is
   Alden's to revisit; nothing here changes it unilaterally.
3. **The permission email (`typesafe-permission-request-email.md`).** Its current
   draft cites "Section 2.3(f) of the TypeSafe Master Customer Agreement" by
   number. Sending it as written would quote a clause back to TypeSafe that their
   own agreement does not contain. **Do not send it unchanged.**

## To do, and what became of it (2026-09-21)

- [x] **Alden:** decide whether to revisit ADR-001 now that the premise is unverified.
      Done 2026-09-21: publish, and send no permission request. The decision is recorded
      in `PUBLICATION_PERMISSION_README.md`, and `founder-decision-2026-09-18.md` now
      carries a correction header pointing at this check.
- [ ] **Kiro:** retry the Wayback comparison when the Internet Archive is back up.
      **Still open.** It has not been retried, so the two explanations under "What is NOT
      established" above are still both open, and nothing in the paper claims otherwise.
- [ ] **Kiro:** re-run this check on publication day (§2, §29) and record the result.
      **Still open by design:** publication has not happened.
- [x] Rewrite the permission email so it asks about benchmark publication generally
      rather than citing §2.3(f). **Superseded:** no email will be sent. The draft ships
      as a record, headed with the reason it must not be sent as written.
