// Mock-agent control-flow tests for .claude/workflows/jev-benchmark.js (no model calls).
// Run on the host: node tests/test_workflow_controlflow.js .claude/workflows/jev-benchmark.js
const fs=require("fs");
const src=fs.readFileSync(process.argv[2],"utf8").replace("export const meta","const meta");
const F=Object.getPrototypeOf(async function(){}).constructor;
const REPO="Poll-The-People/customgpt-jev-eval";
const appr=(gate,o={})=>({gate,exit_code:o.exit??0,raw_output:JSON.stringify(Object.assign({gate,approved:true,url_ok:true,offline:false,author:"adorosario",
  created_at:"2026-09-20T11:00:00Z",updated_at:"2026-09-20T11:00:00Z",request_created_at:"2026-09-20T10:00:00Z",
  url:`https://github.com/${REPO}/issues/1#issuecomment-5`,request_sha:HEAD,approved_sha:gate==="G2A"?"3f9c2ab":null},o.j||{}))});
const HEAD="3f9c2ab7766554433221100aabbccddeeff00112";
const heads=["G0","G1","G2A","G2B","G3","G4","G5A","G5B","G6","G7","G8","FOUNDER"].map(g=>({gate:g,git_head:HEAD}));
const ev=[{claim:"c",proof:"sha256 abc"}];
const PASS={verdict:"PASS",blocking_issues:[],evidence:ev,goals_items_proven:["A","B"]};
async function run(name,args,opt={},expect){
  const calls=[];let x_push="";let execPrompt="",recPrompt="",repPrompt="",verPrompt="";let vn=0;
  const agent=async(p,o)=>{const L=o.label;calls.push(L);
    if(L.startsWith("preflight")) return opt.pre||{ok:true,reason:"",checks:[],stale_or_failed:[],prereq_heads:opt.heads||heads,approvals:(opt.approvals||[])};
    if(L.startsWith("execute")){execPrompt=p;return {run_id:"r1",summary:"s",artifacts:[],commands_run:[],open_issues:[],needs_human:[]}}
    if(L.startsWith("repair")){repPrompt=p;} if(L.startsWith("repair")) return {run_id:"r1",summary:"fixed",artifacts:[],commands_run:[],open_issues:[],needs_human:[]};
    if(L.startsWith("verify")){vn++;verPrompt=p; return opt.verify?opt.verify(vn,L):PASS}
    if(L.startsWith("push-tags")){x_push=p;} if(L.startsWith("push-tags")) return opt.push!==undefined?opt.push:{pushed:true,ls_remote:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/benchmark-protocol-v1\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/benchmark-protocol-v1.1\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/freshground-freeze-v1",local_shas:{"benchmark-protocol-v1":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","benchmark-protocol-v1.1":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","freshground-freeze-v1":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}};
    if(L.startsWith("record")){recPrompt=p;return ("rec" in opt)?opt.rec:{written:true,sha256:"x",ticked:["A","B"],stale_targets_found:[],marked_stale:[],approval_request_url:"u"}}
  };
  const parallel=async t=>Promise.all(t.map(f=>f()));
  const r=await new F("agent","parallel","pipeline","phase","log","args","budget",src)(agent,parallel,null,()=>{},()=>{},args,{});
  const got=r.verdict==="PASS"&&r.status?r.status:(r.status||r.verdict||r.error);
  const ok=expect(r,{calls,execPrompt,recPrompt,repPrompt,verPrompt,pushPrompt:x_push});
  console.log((ok?"ok  ":"BAD ")+name.padEnd(48)+String(got).padEnd(20)+(r.reasons?(" "+r.reasons.join("; ")).slice(0,110):""));
  return ok;
}
(async()=>{let bad=0;const t=async(...a)=>{if(!await run(...a))bad++};
 await t("G0 happy path",{gate:"G0"},{},r=>r.verdict==="PASS");
 await t("G2B with no approvals (critic 8b)",{gate:"G2B"},{},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, preflight ok:true but approval not approved",{gate:"G2B"},{approvals:[appr("G2A",{exit:1,j:{approved:false,reason:"none"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, approval from offline hook",{gate:"G2B"},{approvals:[appr("G2A",{j:{offline:true}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, approval by kiro",{gate:"G2B"},{approvals:[appr("G2A",{j:{author:"kirollosatef"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, edited approval",{gate:"G2B"},{approvals:[appr("G2A",{j:{updated_at:"2026-09-21T00:00:00Z"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, approval before request",{gate:"G2B"},{approvals:[appr("G2A",{j:{created_at:"2026-09-20T09:00:00Z",updated_at:"2026-09-20T09:00:00Z"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, approval missing sha",{gate:"G2B"},{approvals:[appr("G2A",{j:{approved_sha:null}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, wrong-repo url",{gate:"G2B"},{approvals:[appr("G2A",{j:{url:"https://github.com/evil/x/issues/1#issuecomment-1"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, non-JSON raw output",{gate:"G2B"},{approvals:[{gate:"G2A",exit_code:0,raw_output:"approved!"}]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B, stale prerequisite",{gate:"G2B"},{pre:{ok:true,reason:"",checks:[],stale_or_failed:["G2A"],prereq_heads:heads,approvals:[appr("G2A")]}},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G2B valid approval: sha reaches executor+push",{gate:"G2B"},{approvals:[appr("G2A")]},(r,x)=>r.verdict==="PASS"&&x.execPrompt.includes("3f9c2ab")&&x.calls.some(c=>c.startsWith("push-tags")));
 await t("G2B executor told NOT to push the tag",{gate:"G2B"},{approvals:[appr("G2A")]},(r,x)=>x.execPrompt.includes("Do NOT push benchmark-protocol-v1"));
 await t("G2B fails verify: no repair, no push",{gate:"G2B"},{approvals:[appr("G2A")],verify:()=>({verdict:"FAIL",blocking_issues:["x"],evidence:ev,goals_items_proven:[]})},(r,x)=>r.verdict==="FAIL"&&!x.calls.some(c=>c.startsWith("repair")||c.startsWith("push-tags")));
 await t("G2B push unconfirmed -> FAIL",{gate:"G2B"},{approvals:[appr("G2A")],push:{pushed:true,ls_remote:""}},r=>r.verdict==="FAIL");
 await t("G2B push agent null -> FAIL",{gate:"G2B"},{approvals:[appr("G2A")],push:null},r=>r.verdict==="FAIL");
 await t("PASS with blocking issue -> FAIL",{gate:"G0"},{verify:()=>({...PASS,blocking_issues:["hidden"]})},r=>r.verdict==="FAIL");
 await t("empty-string evidence -> FAIL (critic 2c)",{gate:"G0"},{verify:()=>({...PASS,evidence:[{claim:"",proof:""}]})},r=>r.verdict==="FAIL");
 await t("whitespace evidence -> FAIL",{gate:"G0"},{verify:()=>({...PASS,evidence:[{claim:" ",proof:"x"}]})},r=>r.verdict==="FAIL");
 await t("repair then pass",{gate:"G3"},{verify:n=>n<=3?{verdict:"FAIL",blocking_issues:["b"],evidence:ev,goals_items_proven:[]}:PASS},(r,x)=>r.verdict==="PASS"&&x.calls.filter(c=>c.startsWith("repair")).length===1);
 await t("G3 repair prompt carries post-lock policy",{gate:"G3"},{verify:n=>n<=3?{verdict:"FAIL",blocking_issues:["b"],evidence:ev,goals_items_proven:[]}:PASS},(r,x)=>x.repPrompt.includes("RESULTS LOCK")&&x.repPrompt.includes("NEW run_id"));
 await t("G0 repair prompt has NO post-lock policy",{gate:"G0"},{verify:n=>n<=3?{verdict:"FAIL",blocking_issues:["b"],evidence:ev,goals_items_proven:[]}:PASS},(r,x)=>!x.repPrompt.includes("RESULTS LOCK"));
 await t("verify prompt includes gate work text",{gate:"G0"},{},(r,x)=>x.verPrompt.includes("Test-split guard")&&x.verPrompt.includes("## G0:"));
 await t("stale not marked -> STALE_MARKING_FAILED (7c)",{gate:"G0"},{rec:{written:true,sha256:"x",ticked:["A","B"],stale_targets_found:["G1","G5A"],marked_stale:["G1"]}},r=>r.status==="STALE_MARKING_FAILED");
 await t("under-tick warns (7e)",{gate:"G0"},{rec:{written:true,sha256:"x",ticked:["A"],stale_targets_found:[],marked_stale:[]}},r=>r.warnings.some(w=>w.includes("did not tick")));
 await t("over-tick warns",{gate:"G0"},{rec:{written:true,sha256:"x",ticked:["A","B","Z"],stale_targets_found:[],marked_stale:[]}},r=>r.warnings.some(w=>w.includes("not proven")));
 await t("approval-ish item never ticked",{gate:"G0"},{verify:()=>({...PASS,goals_items_proven:["A","APPROVE G1 comment exists"]})},(r,x)=>!x.recPrompt.includes("APPROVE G1 comment exists\"]"));
 await t("record null -> UNRECORDED",{gate:"G0"},{rec:null},r=>r.verdict==="UNRECORDED");
 await t("human gate G1: record posts APPROVAL REQUEST",{gate:"G1"},{},(r,x)=>x.recPrompt.includes("APPROVAL REQUEST G1")&&r.next.includes("WAITING FOR ALDEN"));
 await t("G2A record asks for APPROVE G2A <sha>",{gate:"G2A"},{approvals:[appr("G1")]},(r,x)=>x.recPrompt.includes("APPROVE G2A <that same git_head sha>")&&x.recPrompt.includes("APPROVAL REQUEST G2A <git_head>"));
 await t("G5A only needs G0 (no approval)",{gate:"G5A"},{},r=>r.verdict==="PASS");
 await t("G5B needs G2B+G5A, no human prevs",{gate:"G5B"},{},r=>r.verdict==="PASS");
 await t("G3 next hint lists dependents",{gate:"G3"},{},r=>r.next.includes("G4"));
 await t("stale request sha (rerun without new request)",{gate:"G2A"},{approvals:[appr("G1",{j:{request_sha:"deadbee11"}})]},r=>r.status==="BLOCKED_PREFLIGHT"&&r.reasons.join().includes("does not match"));
 await t("G2A approved sha != request sha",{gate:"G2B"},{approvals:[appr("G2A",{j:{approved_sha:"1234567"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("missing prereq head",{gate:"G2A"},{approvals:[appr("G1")],heads:[]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("G9 without FOUNDER confirm blocked",{gate:"G9"},{},r=>r.status==="BLOCKED_PREFLIGHT"&&r.reasons.join().includes("FOUNDER"));
 await t("G9 with FOUNDER confirm runs",{gate:"G9"},{approvals:[appr("FOUNDER",{j:{request_sha:null,request_created_at:null}})]},r=>r.verdict==="PASS");
 await t("record request line carries git_head",{gate:"G1"},{},(r,x)=>x.recPrompt.includes("APPROVAL REQUEST G1 <git_head>"));
 await t("protocol_tag arg supersedes tag",{gate:"G2B",protocol_tag:"benchmark-protocol-v1.1"},{approvals:[appr("G2A")]},(r,x)=>r.verdict==="PASS"&&x.pushPrompt.includes("git push origin benchmark-protocol-v1.1")&&x.execPrompt.includes("tag benchmark-protocol-v1.1 LOCALLY"));
 await t("G5A pushes freeze tag only after PASS",{gate:"G5A"},{},(r,x)=>x.pushPrompt.includes("freshground-freeze-v1")&&x.execPrompt.includes("Do NOT push freshground-freeze-v1"));
 await t("G5A fail: no tag push",{gate:"G5A"},{verify:()=>({verdict:"FAIL",blocking_issues:["x"],evidence:ev,goals_items_proven:[]})},(r,x)=>!x.calls.some(c=>c.startsWith("push-tags")));
 await t("check_approval item is tickable",{gate:"G2B"},{approvals:[appr("G2A")],verify:()=>({...PASS,goals_items_proven:["`check_approval.sh G2A` passes."]})},(r,x)=>x.recPrompt.includes("check_approval.sh G2A"));
 await t("push: only v1.1 on origin does not confirm v1",{gate:"G2B"},{approvals:[appr("G2A")],push:{pushed:true,ls_remote:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/benchmark-protocol-v1.1",local_shas:{"benchmark-protocol-v1":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}},r=>r.verdict==="FAIL");
 await t("push: sha mismatch vs local",{gate:"G2B"},{approvals:[appr("G2A")],push:{pushed:true,ls_remote:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/benchmark-protocol-v1",local_shas:{"benchmark-protocol-v1":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}},r=>r.verdict==="FAIL");
 await t("self-approval: request posted by adorosario",{gate:"G2A"},{approvals:[appr("G1",{j:{request_author:"adorosario"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("FOUNDER from another repo rejected",{gate:"G9"},{approvals:[appr("FOUNDER",{j:{request_sha:null,url:"https://github.com/evil/x/issues/1#issuecomment-1"}})]},r=>r.status==="BLOCKED_PREFLIGHT");
 await t("max_repair_rounds capped at 2",{gate:"G3",max_repair_rounds:25},{verify:()=>({verdict:"FAIL",blocking_issues:["b"],evidence:ev,goals_items_proven:[]})},(r,x)=>x.calls.filter(c=>c.startsWith("repair")).length===2);
 await t("G2B executor gets full request sha",{gate:"G2B"},{approvals:[appr("G2A")]},(r,x)=>x.execPrompt.includes(HEAD));
 await t("G3 execute prompt carries post-lock policy",{gate:"G3"},{},(r,x)=>x.execPrompt.includes("RESULTS LOCK"));
 await t("unknown gate",{gate:"G2"},{},r=>!!r.error);
 console.log(bad?`${bad} BAD`:"ALL OK"); process.exitCode = bad ? 1 : 0;
})();
