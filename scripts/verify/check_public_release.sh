#!/usr/bin/env bash
# Hard pre-submission gate: the public repository the paper links must actually hold
# what Appendix D says it holds.
#
#   bash scripts/verify/check_public_release.sh [owner/repo]
#
# Appendix D of the paper names artefacts by path and tells a reader to run `ots verify`
# and `scripts/paper_figures.py` against them. Until the export built by
# scripts/build_public_export.sh has been pushed, every one of those paths is a 404 and
# those sentences are false. Pushing is a publication action and belongs to Alden under
# CLAUDE.md rule 12, so this script cannot fix the problem; it exists so that nobody can
# submit the paper without noticing. Exit 0 means the paper's release claims hold.
set -uo pipefail
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

missing=0
for p in "${PATHS[@]}"; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/$p" || echo 000)
  if [ "$code" = "200" ]; then
    echo "  ok    $p"
  else
    echo "  HTTP $code  $p"
    missing=$((missing + 1))
  fi
done

if [ "$missing" -ne 0 ]; then
  cat >&2 <<MSG

FAIL: $missing of ${#PATHS[@]} artefacts the paper names are not in $REPO_SLUG@$BRANCH.
Appendix D and section 12 tell a reader to fetch these. The paper must not be submitted
until they are there. The fix is not an edit to the paper:

  bash scripts/build_public_export.sh ../jev-rag-claim-verification
  # then Alden commits and pushes that repository (CLAUDE.md rule 12: no agent publishes)
MSG
  exit 1
fi
echo "PASS: all ${#PATHS[@]} artefacts the paper names are public at $REPO_SLUG@$BRANCH"
