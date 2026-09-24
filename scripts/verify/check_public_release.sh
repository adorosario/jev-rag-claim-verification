#!/usr/bin/env bash
# Hard pre-submission gate: the public repository the paper links must actually hold
# what Appendix D says it holds, and it must hold the same bytes.
#
#   bash scripts/verify/check_public_release.sh [owner/repo] [branch]
#
# Appendix D of the paper names artefacts by path and tells a reader to run `ots verify`
# and `scripts/paper_figures.py` against them. Until the export built by
# scripts/build_public_export.sh has been pushed, every one of those paths is a 404 and
# those sentences are false. Pushing is a publication action and belongs to Alden under
# CLAUDE.md rule 12, so this script cannot fix the problem; it exists so that nobody can
# submit the paper without noticing. Exit 0 means the paper's release claims hold.
#
# This gate used to ask each path for an HTTP status and nothing else. That is not the
# claim the paper makes. A stale export answers 200 on every path while publishing values
# the paper says were fixed, and one did: for a whole review round configs/models.yaml
# was published with `max_completion_tokens: 16` for gpt-6-astra, the exact cap Appendix C
# calls the bug that wiped 84 of 495 answers, while this gate printed PASS, because the
# export predated the fix by 74 minutes and a 200 cannot tell the two apart. So every path
# is now compared by sha256 against the copy in this checkout. Presence is not correctness.
set -uo pipefail
# Every comparison below reads the local copy of a path, so the script runs from the root of
# the checkout it is comparing against, not from wherever it was invoked. It sits at
# scripts/verify/ in this repository and at scripts/verify/ in the released package, so the
# same two levels up is the right root in both.
cd "$(dirname "$0")/../.." || exit 2
REPO_SLUG="${1:-adorosario/jev-rag-claim-verification}"
BRANCH="${2:-main}"
BASE="https://raw.githubusercontent.com/$REPO_SLUG/$BRANCH"

# Every path the paper names for a reader to go and check. Keep this list in step with
# the check_present list in scripts/build_public_export.sh and with Appendix D;
# tests/test_public_export.py fails if one list gains a path the other lacks.
#
# This script is itself in the released package, so a reader runs this list rather than
# reading ours. It used to name configs/prompts/claim_verification_frontier.txt, a file that has
# never existed in this repository under that name, so one row of this gate could only ever
# report 404 and no build could have fixed it. The prompts are the three files below.
PATHS=(
  arxiv/evidence/raw_hashes.json
  arxiv/evidence/raw_hashes.json.ots
  arxiv/evidence/input_hashes.json
  arxiv/generated/numbers.json
  arxiv/generated/numbers.tex
  configs/benchmark.yaml
  configs/models.yaml
  configs/pricing_2026-09-20.yaml
  configs/prompts/binary_v1.txt
  configs/prompts/jev_grounding_v1.json
  configs/prompts/frontier_response_modes.json
  data/manifests/aggrefact_revision.json
  data/manifests/llm_aggrefact_published_leaderboard.json
  docs/reference/incident-log.md
  legal/PUBLICATION_PERMISSION_README.md
  legal/typesafe-mca-2.3f-excerpt.md
  legal/typesafe-mca-snapshot-2026-09-20.sha256
  legal/founder-decision-2026-09-18.md
  legal/typesafe-permission-request-email.md
  runs/model_live_check.json
  medium/article-v3.md
  medium/story-link.md
  scripts/verify/check_public_release.sh
  scripts/paper_figures.py
  scripts/verify/paired_analysis.py
  scripts/verify/check_paper_macros.py
  scripts/verify/check_article_numbers.py
  src/metrics.py
  src/pricing.py
  src/run_eval.py
  tests/test_metrics.py
  tests/test_crossfit.py
  tests/test_cost.py
  tests/test_paper_evidence.py
  tests/test_evidence_integrity.py
  tests/test_model_registry.py
  runs/explore-jev-dev/jev_dev_sample.jsonl
  runs/paired-jev-20260921/predictions_jev.jsonl
  runs/paired-astra-lowthink-20260921-p2b/predictions_gpt6_astra.jsonl
  runs/paired-astra-highthink-20260921-p2b/predictions_gpt6_astra.jsonl
  runs/paired-minicheck-20260922/predictions_minicheck.jsonl
  runs/paired-astra-vs-jev-20260921-v2/predictions_gpt6_astra.jsonl
  runs/paired-astra-vs-jev-20260921/predictions_gpt6_astra.jsonl
  runs/paired-astra-vs-jev-20260921/run_manifest.json
  runs/paired-astra-highthink-20260921/predictions_gpt6_astra.jsonl
  runs/paired-astra-highthink-20260921/run_manifest.json
)

# The one path that cannot be compared byte for byte. configs/models.yaml is published by
# scripts/_export_models_yaml.py, which removes the private production-verifier entry and
# re-serialises the file, so the published bytes differ from ours by construction. What the
# paper says about it is checked instead: the caps the reported runs sent, the incident that
# changed them, and the reason the one untested cap is still 16. Whitespace is flattened
# first, because the serialiser folds long note strings across lines at its own width.
MODELS_YAML=configs/models.yaml
MODELS_YAML_MUST_SAY=(
  "low: 256"
  "high: 4096"
  "max_completion_tokens: 16 until"
  "never became an arm"
)
# And the published registry describes these five systems and no others. This is the
# positive form of "the private entry was stripped": a list of what the paper evaluates
# says the same thing without this script, which ships, naming what the export withholds.
MODELS_YAML_MODELS="claude_sonnet5 gpt56_sol gpt6_astra jev minicheck"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

tmp=$(mktemp -d) || exit 2
trap 'rm -rf "$tmp"' EXIT

missing=0
differing=0
for p in "${PATHS[@]}"; do
  code=$(curl -sS -o "$tmp/fetched" -w '%{http_code}' "$BASE/$p" || echo 000)
  if [ "$code" != "200" ]; then
    echo "  HTTP $code  $p"
    missing=$((missing + 1))
    continue
  fi
  if [ ! -f "$p" ]; then
    # A path this list names and this checkout does not have cannot be compared, and a
    # comparison that cannot run must not report ok: that is how the status-only version of
    # this gate passed on a stale file. tests/test_public_export.py fails on the same drift.
    echo "  NO LOCAL COPY TO COMPARE  $p"
    differing=$((differing + 1))
    continue
  fi
  if [ "$p" = "$MODELS_YAML" ]; then
    flat=$(tr '\n\t' '  ' < "$tmp/fetched" | tr -s ' ')
    bad=0
    for needle in "${MODELS_YAML_MUST_SAY[@]}"; do
      case "$flat" in *"$needle"*) ;; *) echo "  PUBLISHED COPY DOES NOT SAY \"$needle\"  $p"; bad=1 ;; esac
    done
    listed=$(awk '/^models:/{inside=1; next} inside && /^[^ ]/{inside=0}
                  inside && /^  [A-Za-z0-9_]+:$/{gsub(/[ :]/, ""); print}' "$tmp/fetched" \
             | sort | tr '\n' ' ')
    listed=${listed% }
    if [ "$listed" != "$MODELS_YAML_MODELS" ]; then
      echo "  PUBLISHED COPY LISTS \"$listed\", expected \"$MODELS_YAML_MODELS\"  $p"
      bad=1
    fi
    if [ "$bad" -eq 0 ]; then
      echo "  ok (content)  $p"
    else
      differing=$((differing + 1))
    fi
    continue
  fi
  want=$(sha256_of "$p")
  got=$(sha256_of "$tmp/fetched")
  if [ "$want" = "$got" ]; then
    echo "  ok    $p"
  else
    echo "  STALE  $p  (here $want, published $got)"
    differing=$((differing + 1))
  fi
done

if [ "$missing" -ne 0 ] || [ "$differing" -ne 0 ]; then
  cat >&2 <<MSG

FAIL: of ${#PATHS[@]} artefacts the paper names in $REPO_SLUG@$BRANCH,
$missing are not there and $differing do not match this checkout.
Appendix D and section 12 tell a reader to fetch these, and tell them what they will find.
The paper must not be submitted until the published copies say it. The fix is not an edit
to the paper:

  bash scripts/build_public_export.sh ../jev-rag-claim-verification
  # then Alden commits and pushes that repository (CLAUDE.md rule 12: no agent publishes)
MSG
  exit 1
fi
echo "PASS: all ${#PATHS[@]} artefacts the paper names are public at $REPO_SLUG@$BRANCH"
echo "      and each matches this checkout ($MODELS_YAML by content: the export strips its private entry)"
