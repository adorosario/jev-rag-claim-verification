# Founder decision record: publish without waiting for TypeSafe permission

> **Correction (2026-09-21). The premise of this record did not survive the check it
> assumed.** On 2026-09-20 the live TypeSafe Master Customer Agreement was fetched and
> hashed. It contains no clause restricting the publication of benchmarks or performance
> information: the words "benchmark" and "publish" do not appear in it, and clause 2.3(f)
> is "interfere with the operation of the Services"
> (`typesafe-mca-2.3f-excerpt.md`). The "Known risk" section below is therefore historical
> and is kept unedited as the record of what was believed on 2026-09-18. The operative
> decision is Alden's of 2026-09-21 in `PUBLICATION_PERMISSION_README.md`: publish, and
> send no permission request. The waiver email named below was drafted and never sent; it
> ships in this folder marked as such.

- **Date:** 2026-09-18
- **Decided by:** Alden Do Rosario (Founder & CEO, Poll the People, Inc. / CustomGPT.ai)
- **Recorded by:** Claude Code session that set up the repo, from Alden's instruction.

## Decision
Alden will publish the benchmark (arXiv, Medium, public repo) without waiting for written permission from TypeSafe. This departs from spec §3, which treats permission as a blocking gate. The waiver request email is still to be sent. See [typesafe-permission-request-email.md](typesafe-permission-request-email.md).

## Known risk (recorded for completeness)
MCA §2.3(f) restricts customers from publishing benchmarks or performance information. Publishing without permission may be treated as a breach, and TypeSafe could suspend API access. If access is lost before G3 or G4 finish, the full Jev run could be blocked.

**Mitigation:** finish all Jev inference (G3, G4 and Track B in G5B) before anything is published.

## Effect on the workflow
- Publication (G9) is **not** blocked on TypeSafe permission.
- The disclosure text in the paper and article (G8) must state the real permission status accurately.
- Everything else in the spec is unchanged, including independence, disclosure, and no editorial control for the vendor.

## If TypeSafe explicitly refuses
Publishing after an explicit "no" is a materially different situation from publishing without a reply. If TypeSafe refuses in writing, G9 stops. The release is not prepared until Alden and counsel review it. See decision log ADR-001.

## Confirmation
Alden said this in chat, and a Claude Code session wrote this record. **To do (Alden):** confirm it by commenting "CONFIRM founder-decision-2026-09-18" on the repo's "Gate approvals" issue. That comment is Alden's own attributable confirmation.
