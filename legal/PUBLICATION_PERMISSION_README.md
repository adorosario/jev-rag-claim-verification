# Publication permission: status and record

> **Current status (2026-09-21, and the status at publication).** No permission is needed
> and none is being sought. The clause this file was opened to deal with is not in the live
> TypeSafe agreement: the current MCA was fetched and hashed on 2026-09-20, and its
> clause 2.3(f) is "interfere with the operation of the Services", with the words "benchmark"
> and "publish" appearing nowhere in the document. The dated table below is the historical
> record of what was believed on 2026-09-18, before that check was made; read the
> 2026-09-21 section at the end for the resolution. Nothing here is an admission that this
> work was published against a contractual restriction, because there is no such
> restriction.
>
> Files in this folder that travel with the public release: this README, the clause excerpt,
> the sha256 of the agreement snapshot it was taken from, the founder decision of 2026-09-18
> with its correction header, and the permission-request email that was drafted and never
> sent. The email ships because three of the other files point at it, headed with the reason
> it was never sent and must not be sent as written. The snapshot itself is TypeSafe's
> document and is not republished. Its sha256 identifies **our capture**, taken on
> 2026-09-20 at the URL the excerpt records: the page's own version line moved while we were
> checking it, so re-fetching it will not reproduce those bytes and the hash is not offered
> as something a reader can match. What a reader can do is fetch the live agreement and read
> clause 2.3(f) for themselves.

| Item | Status (2026-09-18) |
|---|---|
| TypeSafe MCA §2.3(f), which restricts publishing benchmarks | Present in the MCA last updated 2026-08-27. See [typesafe-mca-2.3f-excerpt.md](typesafe-mca-2.3f-excerpt.md). |
| Written permission from TypeSafe | **Not received.** The request email is drafted in [typesafe-permission-request-email.md](typesafe-permission-request-email.md). |
| Decision to publish | **Founder decision: publish without waiting for permission.** See [founder-decision-2026-09-18.md](founder-decision-2026-09-18.md). |
| Credits, early access or engineering help from TypeSafe | None so far. Log anything received here, because it must be disclosed (spec §3.3). |

## What to save here
- Every email to and from TypeSafe about this benchmark, saved as `.eml` or pasted as `.md` with the date in the filename.
- Any reply that grants, refuses or conditions permission.
- Anything TypeSafe provides (credits, rate-limit changes, advice), with dates.

## How this affects the paper and article (G8)
The disclosure paragraph must match the facts in this folder when we publish. For example, if permission was never received, the paper must not say "TypeSafe granted permission". TypeSafe may check facts or confidentiality, but it gets no editorial control over results (spec §3.2).

---

## 2026-09-21: closed. No permission is being sought.

Three things changed since this file was written:

1. **The clause we were worried about is not in the live agreement.** Kiro fetched
   and hashed the current TypeSafe MCA on 2026-09-20. Its §2.3(f) is "interfere with
   the operation of the Services", and the words "benchmark" and "publish" do not
   appear anywhere in the document. See `typesafe-mca-2.3f-excerpt.md` in this folder
   and the `.sha256` of the capture beside it. The go/no-go checkpoint that records the
   same finding is an internal document and is not part of this release; the excerpt is
   the released form of it.
2. **Publishing Jev results is now common practice.** LangChain published a Jev
   evaluation with named competitors and per-call costs on 2026-09-19
   (https://www.langchain.com/blog/jev-agent-evals-langsmith), among others.
3. **Founder decision (Alden, 2026-09-21):** we publish. No permission request will
   be sent, so `typesafe-permission-request-email.md` is kept only as a record of
   what was drafted. It must not be sent as written, because it cites a clause number
   that says something else in the live agreement.

**What this does not change:** the paper and the article still state the TypeSafe
relationship accurately — that we are a customer, that no credits or support were
received, and that TypeSafe had no input into the design, the data or the results.
