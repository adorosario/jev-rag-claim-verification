// Mock-agent control-flow tests for .claude/workflows/arxiv-paper.js (no model calls).
// Run on the host: node tests/test_arxiv_workflow.js .claude/workflows/arxiv-paper.js
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8").replace("export const meta", "const meta");
const F = Object.getPrototypeOf(async function () {}).constructor;

const ev = [{ claim: "c", proof: "sha256 abc" }];
const PASS = { verdict: "PASS", blocking_issues: [], non_blocking_issues: [], evidence: ev };
const EXEC = { summary: "s", files_written: ["f"], commands_run: [], needs_human: [], self_reported_problems: [] };

async function run(name, args, opt = {}, expect) {
  const calls = [];
  const prompts = {};
  let vn = 0;
  const agent = async (p, o) => {
    const L = o.label;
    calls.push(L);
    prompts[L] = p;
    if (L.startsWith("preflight")) return "pre" in opt ? opt.pre : { ok: true, reason: "", checks: [] };
    if (L.startsWith("draft") || L.startsWith("voice") || L.startsWith("execute") || L.startsWith("mentor"))
      return "exec" in opt ? opt.exec : EXEC;
    if (L.startsWith("lens")) return opt.lens ? opt.lens(L) : { lens: L.split(":")[1], verdict: "revise", findings: [] };
    if (L.startsWith("refute")) return opt.refute ? opt.refute(L) : { lens: L.split(":")[1], verdicts: [] };
    if (L.startsWith("repair")) return EXEC;
    if (L.startsWith("verify")) { vn++; return opt.verify ? opt.verify(vn, L) : PASS; }
    if (L.startsWith("record")) return "rec" in opt ? opt.rec : { written: true, path: "p", marked_stale: opt.stale || [] };
  };
  const parallel = async (t) => Promise.all(t.map((f) => f()));
  const r = await new F("agent", "parallel", "pipeline", "phase", "log", "args", "budget", src)(
    agent, parallel, null, () => {}, () => {}, args, {});
  const got = r.status || r.verdict || r.error;
  const ok = expect(r, { calls, prompts });
  console.log((ok ? "ok  " : "BAD ") + name.padEnd(52) + String(got).slice(0, 40));
  return ok;
}

const has = (calls, pre) => calls.filter((c) => c.startsWith(pre)).length;

(async () => {
  let bad = 0;
  const t = async (...a) => { if (!(await run(...a))) bad++; };

  // --- routing and guards -------------------------------------------------------
  await t("unknown phase returns an error", { phase: "D99" }, {},
    (r) => /Unknown phase/.test(r.error || ""));
  await t("preflight failure blocks the phase", { phase: "D2" }, { pre: { ok: false, reason: "D1 is stale", checks: [] } },
    (r) => r.status === "BLOCKED_PREFLIGHT" && /stale/.test(r.reason));
  await t("a dead preflight agent blocks", { phase: "D2" }, { pre: null },
    (r) => r.status === "BLOCKED_PREFLIGHT");
  await t("a dead executor fails the phase", { phase: "D1" }, { exec: null },
    (r) => r.status === "EXECUTOR_FAILED");
  await t("a failed record is not silently a pass", { phase: "D1" }, { rec: { written: false, path: "", marked_stale: [] } },
    (r) => r.status === "RECORD_FAILED" && r.verdict === "UNRECORDED");

  // --- verification is not a rubber stamp ---------------------------------------
  await t("PASS with a blocking issue is a FAIL", { phase: "D1" },
    { verify: () => ({ verdict: "PASS", blocking_issues: ["the numbers do not regenerate"], evidence: ev }) },
    (r) => r.verdict === "FAIL");
  await t("PASS with empty evidence is a FAIL", { phase: "D1" },
    { verify: () => ({ verdict: "PASS", blocking_issues: [], evidence: [] }) },
    (r) => r.verdict === "FAIL");
  await t("PASS with whitespace-only proof is a FAIL", { phase: "D1" },
    { verify: () => ({ verdict: "PASS", blocking_issues: [], evidence: [{ claim: "c", proof: "   " }] }) },
    (r) => r.verdict === "FAIL");
  await t("a dropped verifier is a FAIL", { phase: "D1" },
    { verify: (n) => (n % 3 === 0 ? null : PASS) },
    (r) => r.verdict === "FAIL");

  // --- the repair loop is bounded -----------------------------------------------
  await t("repair stops as soon as verification passes", { phase: "D1" },
    { verify: (n) => (n <= 3 ? { verdict: "FAIL", blocking_issues: ["x"], evidence: ev } : PASS) },
    (r) => r.verdict === "PASS" && r.repair_rounds === 1);
  await t("repair is capped at 2 even when asked for 25", { phase: "D1", max_repair_rounds: 25 },
    { verify: () => ({ verdict: "FAIL", blocking_issues: ["x"], evidence: ev }) },
    (r) => r.verdict === "FAIL" && r.repair_rounds === 2);
  await t("the council phase gets no repair rounds", { phase: "D6" },
    { verify: () => ({ verdict: "FAIL", blocking_issues: ["x"], evidence: ev }) },
    (r, c) => r.repair_rounds === 0 && has(c.calls, "repair") === 0);

  // --- the fan-outs actually fan out --------------------------------------------
  await t("D3 drafts five section groups plus the bibliography", { phase: "D3" }, {},
    (r, c) => has(c.calls, "draft:") === 6 && c.calls.includes("draft:bib"));
  await t("D3 tells every section agent to use macros, not digits", { phase: "D3" }, {},
    (r, c) => ["setup", "same-score", "different-errors", "cascade", "close"]
      .every((k) => /newcommand macro/.test(c.prompts["draft:" + k] || "")));
  await t("D4 runs the voice pass and names the prose gate", { phase: "D4" }, {},
    (r, c) => has(c.calls, "voice:") === 1 && /lint_paper_prose\.py/.test(c.prompts["voice:D4"]));
  await t("D4 tells the voice agent an em dash is '---' in LaTeX", { phase: "D4" }, {},
    (r, c) => /"---"/.test(c.prompts["voice:D4"]));
  await t("D6 runs seven lenses, refutes each, then synthesises", { phase: "D6" },
    { lens: (L) => ({ lens: L.split(":")[1], verdict: "revise", findings: [{ id: L.split(":")[1] + "-f", severity: "major", claim: "c", evidence: "e", attack: "a", fix: "f" }] }) },
    (r, c) => has(c.calls, "lens:") === 7 && has(c.calls, "refute:") === 7 && has(c.calls, "mentor:") === 1);

  // --- a surviving blocker has to surface ---------------------------------------
  await t("a surviving blocker is reported out of the council", { phase: "D6" },
    { lens: (L) => ({ lens: L.split(":")[1], verdict: "reject", findings: [{ id: L.split(":")[1] + "-f", severity: "blocker", claim: "fatal", evidence: "e", attack: "a", fix: "f" }] }),
      refute: (L) => ({ lens: L.split(":")[1], verdicts: [{ id: L.split(":")[1] + "-f", survives: true, revised_severity: "blocker", reasoning: "holds" }] }) },
    (r) => (r.self_reported_problems || []).some((p) => /^BLOCKER/.test(p)));
  await t("a refuted blocker does not survive", { phase: "D6" },
    { lens: (L) => ({ lens: L.split(":")[1], verdict: "revise", findings: [{ id: L.split(":")[1] + "-f", severity: "blocker", claim: "c", evidence: "e", attack: "a", fix: "f" }] }),
      refute: (L) => ({ lens: L.split(":")[1], verdicts: [{ id: L.split(":")[1] + "-f", survives: false, revised_severity: "minor", reasoning: "misread" }] }) },
    (r) => !(r.self_reported_problems || []).some((p) => /^BLOCKER/.test(p)));

  // --- human phase and staleness ------------------------------------------------
  await t("D8 waits for both humans and never self-approves", { phase: "D8" }, {},
    (r, c) => /WAITING FOR ALDEN AND KIRO/.test(r.next) &&
      /PAPER APPROVAL REQUEST D8/.test(c.prompts["record:D8"]) &&
      /Do not approve on anyone's behalf/.test(c.prompts["record:D8"]));
  await t("a PASS marks every later phase stale", { phase: "D1" },
    { stale: ["D2", "D3", "D4", "D5", "D6", "D7", "D8"] },
    (r) => r.warnings.length === 0 && r.marked_stale.length === 7);
  await t("an incomplete stale marking raises a warning", { phase: "D1" }, { stale: ["D2"] },
    (r) => r.warnings.length === 1 && /not marked stale/.test(r.warnings[0]));
  await t("a FAIL marks nothing stale", { phase: "D1" },
    { verify: () => ({ verdict: "FAIL", blocking_issues: ["x"], evidence: ev }) },
    (r) => r.marked_stale.length === 0);

  // --- verify_only skips the executor but not the verifiers ---
  await t("verify_only does not redraft D3", { phase: "D3", verify_only: true }, {},
    (r, c) => has(c.calls, "draft:") === 0 && has(c.calls, "verify:") === 3 && r.verdict === "PASS");
  await t("verify_only still fails on a blocking issue", { phase: "D3", verify_only: true },
    { verify: () => ({ verdict: "PASS", blocking_issues: ["still broken"], evidence: ev }) },
    (r) => r.verdict === "FAIL");
  await t("verify_only tells the verifiers to judge the disk", { phase: "D3", verify_only: true }, {},
    (r) => (r.self_reported_problems || []).some((p) => /did not execute the phase work/.test(p)));

  // --- the sources-of-truth contract --------------------------------------------
  await t("every agent is told the article is not a source of truth", { phase: "D1" }, {},
    (r, c) => /NOT a source of truth/.test(c.prompts["execute:D1"]));
  await t("every agent is told no hand-typed numbers", { phase: "D1" }, {},
    (r, c) => /No hand-typed numbers/.test(c.prompts["execute:D1"]));
  await t("D1 preflight knows it has no predecessor", { phase: "D1" }, {},
    (r, c) => /no previous phase/.test(c.prompts["preflight:D1"]));
  await t("D2 preflight requires D1 to be PASS and not stale", { phase: "D2" }, {},
    (r, c) => /arxiv\/phases\/D1\.json/.test(c.prompts["preflight:D2"]) && /not have stale/.test(c.prompts["preflight:D2"]));

  console.log(bad ? `\n${bad} BAD` : "\nALL OK");
  process.exitCode = bad ? 1 : 0;
})();
