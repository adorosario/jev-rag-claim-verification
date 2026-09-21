#!/usr/bin/env bash
# Offline tests for scripts/verify/check_approval.sh. They make no GitHub calls.
# Run on the host: bash tests/test_check_approval.sh
set -uo pipefail
S=scripts/verify/check_approval.sh
U=https://github.com/Poll-The-People/customgpt-jev-eval/issues/1#issuecomment
T=$(mktemp -d); pass=0; failn=0
c() { # args: id user created updated body
  jq -n --arg id "$1" --arg u "$2" --arg c "$3" --arg up "$4" --arg b "$5" --arg url "$U-$1" \
    '{html_url:$url, user:{login:$u}, created_at:$c, updated_at:$up, body:$b}'; }
at() { c "$1" "$2" "$3" "$3" "$4"; }   # comment that was never edited
REQ1=$(at 1 kirollosatef 2026-09-20T10:00:00Z $'APPROVAL REQUEST G1 aaaa1111\nreport: docs/checkpoints/G1-checkpoint-1-report.md')
REQ2A=$(at 2 kirollosatef 2026-09-21T10:00:00Z 'APPROVAL REQUEST G2A 3f9c2ab77')
OK1=$(at 3 adorosario 2026-09-20T11:00:00Z 'APPROVE G1')
check() { # args: name gate expected_exit comments...
  local name=$1 gate=$2 want=$3; shift 3
  printf '%s\n' "$@" | jq -s . > "$T/c.json"
  out=$(APPROVAL_COMMENTS_FILE="$T/c.json" bash $S "$gate"); got=$?
  if [ "$got" = "$want" ]; then pass=$((pass+1)); else failn=$((failn+1)); echo "FAIL: $name (exit $got, want $want): $out"; fi
}
field() { printf '%s\n' "${@:3}" | jq -s . > "$T/c.json"; APPROVAL_COMMENTS_FILE="$T/c.json" bash $S "$1" | jq -r "$2"; }
expect() { if [ "$2" = "$3" ]; then pass=$((pass+1)); else failn=$((failn+1)); echo "FAIL: $1 (got '$2', want '$3')"; fi; }

check "valid G1 approval"                  G1  0 "$REQ1" "$OK1"
check "approval by someone else"           G1  1 "$REQ1" "$(at 3 kirollosatef 2026-09-20T11:00:00Z 'APPROVE G1')"
check "approval before request"            G1  1 "$REQ1" "$(at 3 adorosario 2026-09-20T09:00:00Z 'APPROVE G1')"
check "edited approval"                    G1  1 "$REQ1" "$(c 3 adorosario 2026-09-20T11:00:00Z 2026-09-20T12:00:00Z 'APPROVE G1')"
check "substring, not an exact line"       G1  1 "$REQ1" "$(at 3 adorosario 2026-09-20T11:00:00Z 'not ready to APPROVE G1 yet')"
check "approval for the wrong gate"        G1  1 "$REQ1" "$(at 3 adorosario 2026-09-20T11:00:00Z 'APPROVE G2A 3f9c2ab77')"
check "no request comment"                 G1  1 "$OK1"
check "request without sha is ignored"     G1  1 "$(at 1 kirollosatef 2026-09-20T10:00:00Z 'APPROVAL REQUEST G1')" "$OK1"
check "approval line among other lines"    G1  0 "$REQ1" "$(at 3 adorosario 2026-09-20T11:00:00Z $'looks good\nAPPROVE G1\nthanks')"
check "G1 approval does not approve G10"   G10 1 "$(at 9 kirollosatef 2026-09-20T10:00:00Z 'APPROVAL REQUEST G10 bbbb2222')" "$OK1"
check "rerun: new request voids old OK"    G1  1 "$REQ1" "$OK1" "$(at 5 kirollosatef 2026-09-22T10:00:00Z 'APPROVAL REQUEST G1 cccc3333')"
check "rerun: approval after new request"  G1  0 "$REQ1" "$OK1" "$(at 5 kirollosatef 2026-09-22T10:00:00Z 'APPROVAL REQUEST G1 cccc3333')" "$(at 6 adorosario 2026-09-22T11:00:00Z 'APPROVE G1')"
check "G2A needs a sha"                    G2A 1 "$REQ2A" "$(at 4 adorosario 2026-09-21T11:00:00Z 'APPROVE G2A')"
check "G2A sha matches request (prefix)"   G2A 0 "$REQ2A" "$(at 4 adorosario 2026-09-21T11:00:00Z 'APPROVE G2A 3f9c2ab')"
check "G2A sha differs from request"       G2A 1 "$REQ2A" "$(at 4 adorosario 2026-09-21T11:00:00Z 'APPROVE G2A 1234567')"
check "FOUNDER confirm"                    FOUNDER 0 "$(at 7 adorosario 2026-09-19T08:00:00Z 'CONFIRM founder-decision-2026-09-18')"
check "FOUNDER confirm by someone else"    FOUNDER 1 "$(at 7 kirollosatef 2026-09-19T08:00:00Z 'CONFIRM founder-decision-2026-09-18')"
check "FOUNDER confirm edited"             FOUNDER 1 "$(c 7 adorosario 2026-09-19T08:00:00Z 2026-09-19T09:00:00Z 'CONFIRM founder-decision-2026-09-18')"
check "unknown gate name"                  BOGUS 2 "$OK1"
check "approval inside code fence ignored" G1 1 "$REQ1" "$(at 3 adorosario 2026-09-20T11:00:00Z $'you will need:\n```\nAPPROVE G1\n```\nnot yet')"
check "REVOKE after approval"              G1  1 "$REQ1" "$OK1" "$(at 8 adorosario 2026-09-20T12:00:00Z 'REVOKE G1')"
check "FOUNDER revoked"                    FOUNDER 1 "$(at 7 adorosario 2026-09-19T08:00:00Z 'CONFIRM founder-decision-2026-09-18')" "$(at 8 adorosario 2026-09-19T09:00:00Z 'REVOKE FOUNDER')"
expect "request_author is reported"     "$(field G1 .request_author "$REQ1" "$OK1")" "kirollosatef"
expect "offline hook output is flagged" "$(field G1 .offline "$REQ1" "$OK1")" "true"
expect "request_sha is extracted"       "$(field G1 .request_sha "$REQ1" "$OK1")" "aaaa1111"
expect "approved_sha is extracted"      "$(field G2A .approved_sha "$REQ2A" "$(at 4 adorosario 2026-09-21T11:00:00Z 'APPROVE G2A 3f9c2ab')")" "3f9c2ab"
rm -rf "$T"; echo "check_approval: $pass passed, $failn failed"; [ "$failn" = 0 ]
