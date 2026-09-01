#!/usr/bin/env python3
"""Publish completed podcast ASR workspaces to a redesigned static website.

The site has two product-level zones:

1. Import studio: paste a podcast URL or drop/upload audio (static UI for now).
2. Transcribed library: searchable index of completed ASR episodes.

Each episode page combines official Xiaoyuzhou page context, LLM summary,
metrics, downloads, and searchable transcript pages.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PODCAST_ROOT = Path(os.environ.get("PODCAST_ROOT", "~/podcast")).expanduser()
STATIC_ROOT = Path(os.environ.get("SENSEVOICE_STATIC_ROOT", "~/deployments/sensevoice/static")).expanduser()
LIBRARY_SLUG = "podcast-asr"
LIBRARY_DIR = STATIC_ROOT / LIBRARY_SLUG
DEFAULT_PORT = int(os.environ.get("SENSEVOICE_WEB_PORT", "8020"))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def slugify(text: str, fallback: str) -> str:
    text = (text or "").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", text)
    ascii_text = re.sub(r"-+", "-", ascii_text).strip("-")
    if ascii_text and (re.search(r"[a-z]", ascii_text) or len(ascii_text) >= 8):
        return ascii_text[:88].strip("-") or fallback
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text or re.fullmatch(r"[\u4e00-\u9fff-]+", text):
        return fallback
    return text[:88].strip("-") or fallback


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(float(seconds or 0)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_bytes(n: int | float | None) -> str:
    if not n:
        return "—"
    n = float(n)
    units = ["B", "KB", "MB", "GB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.1f} {units[i]}" if i else f"{int(n)} B"


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def get_lan_ip() -> str:
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        for item in out.split():
            if item and not item.startswith("127."):
                return item
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def artifact_sig(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        if not path.exists():
            continue
        st = path.stat()
        h.update(str(path).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def coverage_from_chunks(trans: dict[str, Any]) -> tuple[float, list[list[float]]]:
    chunks = trans.get("chunks") or []
    intervals = sorted(
        (float(c.get("start", 0)), float(c.get("end", 0)))
        for c in chunks
        if not c.get("error") and c.get("end") is not None
    )
    merged: list[list[float]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + 0.1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    covered = sum(e - s for s, e in merged)
    duration = float(trans.get("duration_seconds") or 0)
    pct = covered / duration * 100 if duration else 0.0
    return pct, merged


@dataclass
class Episode:
    work_dir: Path
    output_dir: Path
    trans_path: Path
    trans: dict[str, Any]
    benchmark_path: Path | None
    benchmark: dict[str, Any]
    manifest_path: Path | None
    site_meta: dict[str, Any]
    page_context: dict[str, Any]
    slug: str
    device: str
    artifacts: dict[str, Path] = field(default_factory=dict)
    signature: str = ""

    @property
    def dest_dir(self) -> Path:
        return LIBRARY_DIR / self.slug

    @property
    def legacy_dir(self) -> Path:
        return STATIC_ROOT / self.slug


def discover_episodes(root: Path = PODCAST_ROOT) -> list[Episode]:
    episodes: list[Episode] = []
    for output_dir in sorted(root.glob("*/output")):
        trans_candidates = sorted(output_dir.glob("transcription_*.json"))
        if not trans_candidates:
            continue
        def _ok_count(path: Path) -> int:
            data = read_json(path, {}) or {}
            summary = data.get("summary") or {}
            if "ok_chunks" in summary:
                return int(summary.get("ok_chunks") or 0)
            return sum(1 for c in data.get("chunks") or [] if not c.get("error"))
        trans_path = max(trans_candidates, key=lambda p: (_ok_count(p), p.stat().st_mtime))
        trans = read_json(trans_path, {}) or {}
        if not trans.get("chunks"):
            continue
        bench_candidates = sorted(output_dir.glob("benchmark_cpu_gpu_*.json"), key=lambda p: p.stat().st_mtime)
        benchmark_path = bench_candidates[-1] if bench_candidates else None
        benchmark = read_json(benchmark_path, {}) if benchmark_path else {}
        manifest_path = output_dir / "manifest.json" if (output_dir / "manifest.json").exists() else None
        site_meta = read_json(output_dir / "site_meta.json", {}) or {}
        page_context = read_json(output_dir / "episode_page_context.json", {}) or {}
        episode_id = str(trans.get("episode_id") or output_dir.parent.name)
        fallback = f"xiaoyuzhou-{episode_id}" if "xiaoyuzhou" in output_dir.parent.name else episode_id
        title = page_context.get("official_title") or trans.get("title") or episode_id
        slug = str(site_meta.get("slug") or slugify(title, fallback))
        device = str(trans.get("device") or trans_path.stem.replace("transcription_", "") or "cuda")
        artifacts: dict[str, Path] = {f"transcription_{device}.json": trans_path}
        for suffix in ["md", "txt", "srt"]:
            p = output_dir / f"transcript_{device}.{suffix}"
            if p.exists():
                artifacts[p.name] = p
        for name in ["manifest.json", "site_meta.json", "podcast_summary.json", "podcast_summary.md", "tldr_infographic.png", "tldr_infographic_base.png", "tldr_infographic_meta.json", "tldr_infographic_prompt.txt", "episode_page_context.json", "episode_page_context.md"]:
            p = output_dir / name
            if p.exists():
                artifacts[name] = p
        if benchmark_path:
            artifacts[benchmark_path.name] = benchmark_path
        zip_candidates = sorted(output_dir.glob("*_asr_results.zip"), key=lambda p: p.stat().st_mtime)
        if zip_candidates:
            artifacts["asr_results.zip"] = zip_candidates[-1]
        ep = Episode(
            work_dir=output_dir.parent,
            output_dir=output_dir,
            trans_path=trans_path,
            trans=trans,
            benchmark_path=benchmark_path,
            benchmark=benchmark,
            manifest_path=manifest_path,
            site_meta=site_meta,
            page_context=page_context,
            slug=slug,
            device=device,
            artifacts=artifacts,
        )
        ep.signature = artifact_sig(list(artifacts.values()))
        episodes.append(ep)
    return episodes


CSS = """
:root{color-scheme:light;--bg:#f6f7f8;--panel:#fff;--panel2:#eef0f2;--ink:#1b2126;--muted:#55606a;--line:#dde1e5;--line-strong:#c3c9cf;--blue:#0b6bcb;--green:#157f3d;--amber:#9a6a00;--red:#c22828}*{box-sizing:border-box}html{scroll-behavior:smooth;background:var(--bg)}body{margin:0;background:var(--bg);color:var(--ink);font:14px/21px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif}.wrap,.shell{max-width:1120px;margin:0 auto;padding:24px 24px 48px}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}button,input,select{font:inherit}button,a{touch-action:manipulation}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--blue);outline-offset:2px}.appbar{position:sticky;top:0;z-index:10;min-height:56px;background:var(--panel);border-bottom:1px solid var(--line)}.appbar-inner{max-width:1120px;min-height:56px;margin:0 auto;padding:8px 24px;display:flex;align-items:center;justify-content:space-between;gap:12px}.brand{font-weight:600;line-height:18px}.brand small{display:block;color:var(--muted);font-size:12px;font-weight:400}.nav,.actions,.downloads,.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.nav a{color:var(--muted);font-size:13px}.nav a[aria-current="page"]{color:var(--ink);font-weight:600}.dim,.muted{color:var(--muted)}.page-header{margin:24px 0}.page-header h1{font-size:24px;line-height:30px;margin:0 0 4px;font-weight:600}.page-header p{margin:0;color:var(--muted)}h2{font-size:20px;line-height:28px;margin:0;font-weight:600}h3{font-size:16px;line-height:24px;margin:0;font-weight:600}.panel,.card,.section{background:var(--panel);border:1px solid var(--line);border-radius:6px}.panel,.section{padding:24px}.btn{min-height:32px;border:1px solid var(--line-strong);border-radius:6px;background:var(--panel);color:var(--ink);padding:5px 12px;font-weight:600;display:inline-flex;align-items:center;justify-content:center;cursor:pointer}.btn.primary{border-color:var(--blue);background:var(--blue);color:#fff}.btn:disabled{opacity:.6;cursor:not-allowed}.import{display:grid;gap:16px;padding:16px}.section-head{display:flex;align-items:center;justify-content:space-between;gap:16px}.section-head h2{font-size:20px}.section-label{color:var(--muted);font-size:13px}.tabs{display:flex;border-bottom:1px solid var(--line)}.tab{height:36px;border:0;border-bottom:2px solid transparent;background:transparent;color:var(--muted);padding:0 16px;cursor:pointer}.tab.active{border-color:var(--blue);color:var(--ink);font-weight:600}.field{display:flex;gap:8px}.field input,.search{min-width:0;height:40px;border:1px solid var(--line-strong);border-radius:6px;background:var(--panel);color:var(--ink);padding:0 12px;flex:1}.field .btn{height:40px;min-height:40px}.drop{border:1px dashed var(--line-strong);border-radius:6px;background:var(--panel2);padding:16px}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.step{padding:12px;border-right:1px solid var(--line);font-size:12px}.step:last-child{border-right:0}.step strong{display:block;font-size:13px}.task-status{padding:16px;background:var(--panel2);border:1px solid var(--line);border-radius:4px}.task-status .progress{margin-top:12px}.progress{height:4px;overflow:hidden;background:var(--line);border-radius:4px}.progress span{display:block;width:36%;height:100%;background:var(--blue);animation:progress-slide 1.25s ease-in-out infinite}@keyframes progress-slide{0%{transform:translateX(-120%)}100%{transform:translateX(320%)}}@media(prefers-reduced-motion:reduce){.progress span{animation:none;transform:none}}.stats,.metrics{display:grid;grid-template-columns:repeat(4,1fr);margin:24px 0;border:1px solid var(--line);border-radius:6px;background:var(--panel)}.stat,.metric{padding:12px 16px;border-right:1px solid var(--line)}.stat:last-child,.metric:last-child{border-right:0}.stat small,.metric small{display:block;color:var(--muted);font-size:12px}.stat strong,.metric strong{display:block;font-size:20px;line-height:28px;font-weight:600}.library-head{margin:24px 0 12px}.library-toolbar{display:flex;gap:12px;align-items:center}.library-toolbar .search{max-width:360px}.live-count{color:var(--muted);font-size:13px;white-space:nowrap}.grid,.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.episode,.card{padding:16px;display:grid;gap:12px}.episode h3,.card-title{font-size:18px;line-height:26px}.meta{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted);font-size:13px}.meta span+span:before,.metric-line span+span:before{content:"·";margin-right:8px;color:var(--line-strong)}.summary{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:3;overflow:hidden;margin:0;color:var(--muted)}.metric-line{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:8px 0;color:var(--muted);font-size:12px}.empty-state{padding:24px 0;color:var(--muted)}.episode{scroll-margin-top:72px}.episode:target{outline:2px solid var(--blue);outline-offset:2px}.episode-hero{margin:24px 0 0}.episode-hero .metrics{margin:16px 0}.episode-layout{display:grid;grid-template-columns:240px minmax(0,1fr);gap:24px;margin-top:24px}.toc{position:sticky;top:72px;align-self:start;padding:12px}.toc a{display:block;color:var(--muted);padding:8px 4px}.content{display:grid;gap:16px}.content .panel{scroll-margin-top:72px}.outline{display:grid}.outline-row,.topic-block{padding:12px 0;border-bottom:1px solid var(--line)}.outline-row:last-child,.topic-block:last-child{border-bottom:0}.outline-row{display:grid;grid-template-columns:72px 1fr;gap:12px}.time{color:var(--blue);font-weight:600}.quote{border-left:3px solid var(--blue);background:var(--panel2);padding:12px;margin:12px 0}.summary-grid,.compare,.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:4px}table{width:100%;min-width:640px;border-collapse:collapse}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600}.chunk{max-width:76ch;padding:16px 0;border:0;border-bottom:1px solid var(--line);border-radius:0;background:transparent;margin:0}.chunk-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}.chunk p{margin:0;white-space:pre-wrap}.transcript-toolbar{position:sticky;top:56px;z-index:9;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--line)}.transcript-preview{max-height:420px;overflow:auto;padding:16px;border:1px solid var(--line);border-radius:4px;background:var(--panel2)}.footer{margin-top:32px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}mark{background:#ffe8a3;color:var(--ink);padding:0 2px}.toast{position:fixed;right:24px;bottom:24px;z-index:20;max-width:360px;padding:12px 16px;background:var(--panel);border:1px solid var(--line-strong);border-radius:6px;display:none}.toast.show{display:flex;gap:12px;align-items:start}.toast button{margin-left:auto;border:0;background:transparent;color:var(--muted);cursor:pointer;font-size:18px}@media(max-width:959px){.stats,.metrics{grid-template-columns:repeat(2,1fr)}.stat:nth-child(2),.metric:nth-child(2){border-right:0}.stat:nth-child(-n+2),.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}.episode-layout{grid-template-columns:1fr}.toc{position:sticky;top:56px;z-index:9;display:flex;gap:12px;overflow-x:auto;white-space:nowrap;background:var(--bg);border-bottom:1px solid var(--line);border-radius:0;padding:8px 0}.toc a{padding:4px 0}.content{margin-top:8px}}@media(max-width:599px){.wrap,.shell{padding:16px 12px 32px}.appbar-inner{padding:8px 12px;align-items:flex-start}.nav{justify-content:flex-end}.page-header{margin:16px 0}.panel,.section{padding:16px}.field{flex-direction:column}.field .btn{width:100%}.pipeline{grid-template-columns:1fr}.step{border-right:0;border-bottom:1px solid var(--line)}.step:last-child{border-bottom:0}.grid,.cards,.summary-grid,.compare,.grid2{grid-template-columns:1fr}.library-toolbar{align-items:stretch;flex-wrap:wrap}.library-toolbar .search{max-width:none;width:100%}.toast{right:12px;bottom:12px;left:12px;max-width:none}.transcript-toolbar{top:56px}.outline-row{grid-template-columns:72px minmax(0,1fr)}}@media(max-width:359px){.appbar-inner{flex-wrap:wrap}.nav{padding-bottom:4px}}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#101315;--panel:#171b1e;--panel2:#0c0e10;--ink:#e8ebee;--muted:#9aa5ae;--line:#2b3238;--line-strong:#3d454d;--blue:#6ea8fe;--green:#4ecb7f;--amber:#e3b341;--red:#f08080}mark{background:#604d13;color:var(--ink)}}
""".strip()

SEARCH_JS = r"""
const searchBox=document.getElementById('searchBox');const clearSearch=document.getElementById('clearSearch');const matchCount=document.getElementById('matchCount');const chunks=[...document.querySelectorAll('.chunk')];const original=new Map(chunks.map(c=>[c.id,c.querySelector('p').textContent]));function escReg(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}function applySearch(){const q=(searchBox?.value||'').trim();let shown=0;const re=q?new RegExp(escReg(q),'gi'):null;chunks.forEach(c=>{const text=original.get(c.id)||'';const hit=!q||text.toLowerCase().includes(q.toLowerCase());c.style.display=hit?'':'none';if(hit)shown++;const p=c.querySelector('p');p.innerHTML=re?text.replace(re,m=>`<mark>${m}</mark>`):text});if(matchCount)matchCount.textContent=q?`${shown} 段匹配`:`${chunks.length} 段`}if(searchBox){searchBox.addEventListener('input',applySearch)}if(clearSearch){clearSearch.addEventListener('click',()=>{searchBox.value='';applySearch();searchBox.focus()})}
""".strip()

INDEX_JS = r"""
const tabs=[...document.querySelectorAll('.tab')], urlPane=document.getElementById('urlPane'), filePane=document.getElementById('filePane');
function selectTab(tab){tabs.forEach(t=>{const active=t===tab;t.classList.toggle('active',active);t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1});const url=tab.dataset.tab==='url';urlPane.hidden=!url;filePane.hidden=url}tabs.forEach((t,index)=>{t.onclick=()=>selectTab(t);t.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[next].focus();selectTab(tabs[next])})});
const run=document.getElementById('mockRun'), episodeUrl=document.getElementById('episodeUrl'), taskStatus=document.getElementById('taskStatus'), toast=document.getElementById('toast');
const ACTIVE_TASK_KEY='podcastAsr.activeTaskId', LAST_TASK_KEY='podcastAsr.lastTaskId', LAST_URL_KEY='podcastAsr.lastEpisodeUrl', READY_REFRESHED_TASK_KEY='podcastAsr.readyRefreshedTaskId', COMPLETED_REFRESHED_TASK_KEY='podcastAsr.completedRefreshedTaskId';
let pollTimer=null, activeTask=null;
function escHtml(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function parseEpisodeId(url){const m=String(url||'').match(/(?:\/episode\/|^)([0-9a-fA-F]{12,})(?:[/?#].*)?$/);return m?m[1]:''}
function showToast(title,msg){if(!toast)return;toast.innerHTML=`<div><strong>${escHtml(title)}</strong><br><span class="dim">${escHtml(msg||'')}</span></div><button type="button" aria-label="关闭提示">×</button>`;toast.classList.add('show');toast.querySelector('button').onclick=()=>toast.classList.remove('show');setTimeout(()=>toast.classList.remove('show'),4200)}
function setTaskStatus(html){if(!taskStatus)return;taskStatus.style.display='block';taskStatus.innerHTML=html}
function setRunState(live){if(!run)return;run.disabled=!!live;run.textContent=live?'运行中…':'创建任务'}
function rememberTask(task,url){if(task?.job_id){localStorage.setItem(LAST_TASK_KEY,task.job_id);if(['queued','running'].includes(task.status||''))localStorage.setItem(ACTIVE_TASK_KEY,task.job_id);else localStorage.removeItem(ACTIVE_TASK_KEY)}if(url)localStorage.setItem(LAST_URL_KEY,url)}
function taskPhase(task){if(task.asr_chunks&&Number(task.asr_ok_chunks||0)<Number(task.asr_chunks))return `x570 正在分片转写：${task.asr_ok_chunks||0}/${task.asr_chunks}`;if(!task.has_transcription)return '正在准备音频并启动 x570 ASR';if(!task.has_summary)return 'ASR 已完成，正在生成播客总结';if(!task.has_tldr_image)return '转写和总结已完成，正在生成 TLDR 图';if(['queued','running'].includes(task.status))return '报告已生成，正在导出和发布收尾';return task.message||'任务已完成'}
function taskHtml(task){const labels={queued:'排队中',running:'运行中',completed:'已完成',failed:'失败',unknown:'状态未知'};const status=task.status||'unknown';const live=status==='queued'||status==='running';const asr=task.asr_chunks?`${escHtml(task.asr_ok_chunks)}/${escHtml(task.asr_chunks)} chunks${task.asr_device?' · '+escHtml(String(task.asr_device).toUpperCase()):''}`:(task.has_transcription?'已有':'等待');const links=task.report_url?`<div class="actions"><a class="btn primary" href="${escHtml(task.report_url)}" target="_blank" rel="noopener">打开报告</a><a class="btn" href="${escHtml(task.full_text_url||task.report_url)}" target="_blank" rel="noopener">全文</a><a class="btn" href="${escHtml(task.summary_url||task.report_url)}" target="_blank" rel="noopener">总结</a>${task.has_tldr_image&&task.tldr_image_url?`<a class="btn" href="${escHtml(task.tldr_image_url)}" target="_blank" rel="noopener">TLDR 图</a>`:''}</div>`:'';const log=task.log_tail?`<details><summary class="dim">日志尾部</summary><pre class="transcript-preview">${escHtml(task.log_tail)}</pre></details>`:'';const leaveHint=live?'<p class="dim">可以关闭或离开页面；任务在服务器后台继续运行，回来会自动恢复状态。</p>':'';return `<strong>${escHtml(labels[status]||status)}</strong><div class="dim">任务 ID: ${escHtml(task.job_id)}${task.pid?' · PID '+escHtml(task.pid):''}</div><div>${escHtml(taskPhase(task))}</div><div class="dim">转写: ${asr} · 总结: ${task.has_summary?'已有':'等待'} · TLDR 图: ${task.has_tldr_image?'已有':'等待'}</div>${live?'<div class="progress" aria-label="任务进行中"><span></span></div>':''}${leaveHint}${links}${log}`}
function reloadToEpisode(task,phase){const key=phase==='completed'?COMPLETED_REFRESHED_TASK_KEY:READY_REFRESHED_TASK_KEY;if(localStorage.getItem(key)===task.job_id)return false;localStorage.setItem(key,task.job_id);const next=new URL(location.href);next.searchParams.set('task',task.job_id);next.hash=`episode-${task.episode_id}`;location.replace(next.toString());return true}
async function reportIsReady(task){if(!task.report_url||!task.has_transcription||!task.has_summary)return false;try{const res=await fetch(task.report_url,{method:'HEAD',cache:'no-store'});return res.ok}catch(e){return false}}
async function pollTask(jobId){const res=await fetch(`/api/podcast-asr/tasks/${encodeURIComponent(jobId)}`,{cache:'no-store'});const data=await res.json();if(!res.ok||!data.ok)throw new Error(data.detail||'状态查询失败');const task=data.task;rememberTask(task);setTaskStatus(taskHtml(task));const live=['queued','running'].includes(task.status);setRunState(live);if(live){activeTask=task.job_id;localStorage.setItem(ACTIVE_TASK_KEY,task.job_id);if(await reportIsReady(task)&&reloadToEpisode(task,'ready'))return task;return task}const wasActive=activeTask===task.job_id;if(pollTimer)clearInterval(pollTimer);pollTimer=null;activeTask=null;localStorage.removeItem(ACTIVE_TASK_KEY);if(task.status==='completed'){showToast('任务完成','已发布到播客索引，可打开报告查看。');if(wasActive&&reloadToEpisode(task,'completed'))return task}else if(['failed','unknown'].includes(task.status))showToast('任务未正常完成',task.message||'请查看日志。');return task}
function startPolling(jobId){if(!jobId)return;if(pollTimer)clearInterval(pollTimer);activeTask=jobId;setRunState(true);pollTask(jobId).catch(e=>setTaskStatus(`<strong>状态查询失败</strong><br>${escHtml(e.message)}`));pollTimer=setInterval(()=>pollTask(jobId).catch(e=>setTaskStatus(`<strong>状态查询失败</strong><br>${escHtml(e.message)}`)),5000)}
async function restoreTaskPanel(){const requestedTask=new URLSearchParams(location.search).get('task');const storedActive=localStorage.getItem(ACTIVE_TASK_KEY);const storedLast=localStorage.getItem(LAST_TASK_KEY);const lastUrl=localStorage.getItem(LAST_URL_KEY);if(episodeUrl&&lastUrl&&!episodeUrl.value)episodeUrl.value=lastUrl;if(requestedTask){startPolling(requestedTask);return}if(storedActive){startPolling(storedActive);return}if(storedLast){try{await pollTask(storedLast);return}catch(e){localStorage.removeItem(LAST_TASK_KEY)}}try{const res=await fetch('/api/podcast-asr/tasks',{cache:'no-store'});const data=await res.json();if(res.ok&&data.ok&&Array.isArray(data.tasks)){const task=data.tasks.find(t=>['queued','running'].includes(t.status))||data.tasks[0];if(task?.job_id){rememberTask(task);if(['queued','running'].includes(task.status))startPolling(task.job_id);else setTaskStatus(taskHtml(task))}}}catch(e){}}
if(run){run.onclick=async()=>{const url=(episodeUrl?.value||'').trim();if(!url){setTaskStatus('<strong>请输入小宇宙 episode 链接</strong>');episodeUrl?.focus();return}if(pollTimer)clearInterval(pollTimer);setRunState(true);activeTask=null;localStorage.setItem(LAST_URL_KEY,url);setTaskStatus('<strong>正在创建后台任务…</strong><br><span class="dim">创建成功后可以关闭页面，后台会继续运行。</span>');try{const res=await fetch('/api/podcast-asr/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});const data=await res.json();if(!res.ok||!data.ok)throw new Error(data.detail||data.error||'创建任务失败');const task=data.task;rememberTask(task,url);showToast(data.already_running?'任务已在运行':'后台任务已启动',task.message||'正在运行 pipeline');setTaskStatus(taskHtml(task));startPolling(task.job_id)}catch(e){activeTask=null;localStorage.removeItem(ACTIVE_TASK_KEY);setRunState(false);showToast('创建失败',e.message);setTaskStatus(`<strong>创建失败</strong><br><span>${escHtml(e.message)}</span>`)}}}
document.querySelectorAll('.episode').forEach(card=>{const source=[...card.querySelectorAll('a')].find(link=>link.href.includes('xiaoyuzhoufm.com/episode/'));const episodeId=parseEpisodeId(source?.href||'');if(episodeId){card.id=`episode-${episodeId}`;card.dataset.episodeId=episodeId}});if(location.hash){document.getElementById(location.hash.slice(1))?.scrollIntoView({block:'center'})}
restoreTaskPanel();
const filter=document.getElementById('filter'), shownCount=document.getElementById('shownCount'), noResults=document.getElementById('noResults');if(filter){const applyFilter=()=>{const q=filter.value.toLowerCase().trim();let shown=0;document.querySelectorAll('.episode').forEach(card=>{const match=card.dataset.text.toLowerCase().includes(q)||card.innerText.toLowerCase().includes(q);card.hidden=!match;if(match)shown++});if(shownCount)shownCount.textContent=shown;if(noResults)noResults.hidden=shown>0||!q};filter.oninput=applyFilter;document.getElementById('clearFilter')?.addEventListener('click',()=>{filter.value='';applyFilter();filter.focus()})}
""".strip()


def render_summary_list(items: list[Any]) -> str:
    if not items:
        return ""
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def render_outline_html(outline: list[Any]) -> str:
    if not outline:
        return '<p class="dim">暂无官方 OUTLINE。后续导入小宇宙链接时会自动抓取 show notes。</p>'
    rows: list[str] = []
    for item in outline:
        if isinstance(item, dict):
            notes = "".join(f"<li>{esc(note)}</li>" for note in safe_list(item.get("notes"))[:8])
            rows.append(f'<div class="outline-row"><div class="time">{esc(item.get("timestamp", ""))}</div><div><strong>{esc(item.get("title", ""))}</strong>{f"<ul>{notes}</ul>" if notes else ""}</div></div>')
        else:
            rows.append(f'<div class="outline-row"><div class="time">—</div><div>{esc(item)}</div></div>')
    return '<div class="outline">' + ''.join(rows) + '</div>'


def render_llm_summary_html(summary_data: dict[str, Any], tldr_image: str = "") -> str:
    if not summary_data:
        return '<p class="dim">尚未生成总结。运行 generate_podcast_summary.py 后会显示在这里。</p>'
    guests = []
    for g in safe_list(summary_data.get("guests")):
        if isinstance(g, dict):
            guests.append(f"<li><strong>{esc(g.get('name', 'unknown'))}</strong>：{esc(g.get('role', 'unknown'))}<br><span class=\"dim\">依据：{esc(g.get('evidence', 'unknown'))}</span></li>")
        else:
            guests.append(f"<li>{esc(g)}</li>")
    topics = []
    for i, t in enumerate(safe_list(summary_data.get("topic_summary")), 1):
        if isinstance(t, dict):
            topics.append(f"<div class=\"topic-block\"><h3>{i}. {esc(t.get('topic', '未命名话题'))}</h3><p class=\"dim\">时间：{esc(t.get('timestamp_range', 'unknown'))}</p><p>{esc(t.get('summary', ''))}</p>{render_summary_list(safe_list(t.get('key_points')))}</div>")
        else:
            topics.append(f"<div class=\"topic-block\"><h3>{i}. {esc(t)}</h3></div>")
    quotes = []
    for q in safe_list(summary_data.get("golden_quotes")):
        if isinstance(q, dict):
            quotes.append(f"<div class=\"quote\"><strong>{esc(q.get('quote', ''))}</strong><br><span class=\"dim\">{esc(q.get('speaker_or_context', 'unknown'))} · {esc(q.get('why_it_matters', ''))}</span></div>")
        else:
            quotes.append(f"<div class=\"quote\"><strong>{esc(q)}</strong></div>")
    terms = []
    for term in safe_list(summary_data.get("entities_and_terms")):
        if isinstance(term, dict):
            terms.append(f"<li><strong>{esc(term.get('term', ''))}</strong>：{esc(term.get('explanation', ''))}</li>")
        else:
            terms.append(f"<li>{esc(term)}</li>")
    if tldr_image:
        tldr_html = f'''<figure class="tldr-figure" style="margin:0"><a href="{esc(tldr_image)}" target="_blank" rel="noopener"><img src="{esc(tldr_image)}" alt="TLDR 信息图" loading="eager" style="width:100%;border-radius:18px;border:1px solid var(--line);box-shadow:0 18px 48px rgba(0,0,0,.32);background:#07101a"></a><figcaption class="dim" style="margin-top:8px">TLDR 信息图：原生 GPT Image2 直接生成完整图片（含中文文字）。点击查看原图。</figcaption></figure>'''
    else:
        tldr_html = f"<p>{esc(summary_data.get('tldr', ''))}</p>"
    return f'''<div class="summary-grid"><div><h3>主题</h3><p>{esc(summary_data.get('theme', ''))}</p><h3>嘉宾</h3><ul>{''.join(guests)}</ul></div><div><h3>背景</h3><p>{esc(summary_data.get('background', ''))}</p><h3>TL;DR</h3>{tldr_html}</div></div><h3>讨论的话题总结</h3>{''.join(topics)}<h3>金句</h3>{''.join(quotes)}<div class="summary-grid"><div><h3>关键洞察</h3>{render_summary_list(safe_list(summary_data.get('key_takeaways')))}</div><div><h3>专名与术语</h3><ul>{''.join(terms)}</ul></div></div><h3>注意事项 / ASR 不确定处</h3>{render_summary_list(safe_list(summary_data.get('caveats')))}'''


def copy_artifacts(ep: Episode, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in ep.artifacts.items():
        if src.exists():
            shutil.copy2(src, dest / name)


def render_episode(ep: Episode) -> tuple[str, str, dict[str, Any]]:
    trans = ep.trans
    bench = ep.benchmark or {}
    summary = trans.get("summary") or {}
    chunks = trans.get("chunks") or []
    ok_chunks = [c for c in chunks if not c.get("error")]
    failed_chunks = [c for c in chunks if c.get("error")]
    coverage, merged = coverage_from_chunks(trans)
    speedups = bench.get("speedups") or {}
    devices = bench.get("devices") or {}
    cpu = devices.get("cpu") or {}
    gpu = devices.get("cuda") or devices.get("gpu") or {}
    page_ctx = ep.page_context or {}
    llm_summary = read_json(ep.output_dir / "podcast_summary.json", {}) or {}
    title = page_ctx.get("official_title") or llm_summary.get("title") or trans.get("title") or ep.slug
    podcast_title = page_ctx.get("podcast_title") or llm_summary.get("podcast") or ""
    source_url = page_ctx.get("source_url") or trans.get("url") or ""
    published_time = page_ctx.get("published_time") or llm_summary.get("published_time") or ""
    description = page_ctx.get("description") or ""
    outline = page_ctx.get("outline") or llm_summary.get("official_outline") or []
    duration_seconds = float(trans.get("duration_seconds") or summary.get("audio_seconds") or 0)
    transcript_md = f"transcript_{ep.device}.md"
    transcript_txt = f"transcript_{ep.device}.txt"
    transcript_srt = f"transcript_{ep.device}.srt"
    transcription_json = f"transcription_{ep.device}.json"
    bench_name = ep.benchmark_path.name if ep.benchmark_path else ""
    published_at = datetime.now().isoformat(timespec="seconds")
    llm_tldr = str(llm_summary.get("tldr") or "")

    metric_cards = [
        ("ASR 覆盖率", f"{coverage:.1f}%", f"{len(ok_chunks)}/{len(chunks)} chunks OK"),
        (f"{ep.device.upper()} pipeline", f"{float(summary.get('pipeline_wall_seconds') or summary.get('wall_seconds') or 0):.1f}s", f"{summary.get('x_realtime_wall', '—')}× realtime"),
        ("文本规模", f"{int(summary.get('chars') or sum(int(c.get('chars') or 0) for c in chunks)):,}", "characters"),
        ("CPU/GPU", f"{float(speedups.get('cuda_vs_cpu_inference') or 0):.2f}×", "inference speedup"),
    ]
    metrics_html = "".join(f'<div class="metric"><small>{esc(k)}</small><strong>{esc(v)}</strong><span class="dim">{esc(note)}</span></div>' for k, v, note in metric_cards)

    chunk_rows = []
    transcript_sections = []
    for c in chunks:
        idx = int(c.get("chunk_index") or 0)
        text = c.get("text") or ""
        plain = re.sub(r"\s+", " ", text).strip()
        status = "OK" if not c.get("error") else str(c.get("error"))
        chunk_rows.append(f'<tr><td><a href="full.html#chunk-{idx:03d}">{idx+1:02d}</a></td><td>{esc(c.get("start_ts"))}–{esc(c.get("end_ts"))}</td><td>{float(c.get("duration_seconds") or 0):.1f}s</td><td>{int(c.get("chars") or 0):,}</td><td>{float(c.get("inference_seconds") or 0):.2f}s</td><td>{float(c.get("rtf_inference") or 0):.4f}</td><td>{esc(status)}</td></tr>')
        transcript_sections.append(f'<article class="chunk" id="chunk-{idx:03d}" data-text="{esc(plain)}"><div class="chunk-head"><span class="pill">Chunk {idx+1:02d}</span><h3>{esc(c.get("start_ts"))} → {esc(c.get("end_ts"))}</h3><span class="pill">{int(c.get("chars") or 0):,} 字符</span></div><p>{esc(text)}</p></article>')

    source_link = f'<a class="btn" href="{esc(source_url)}" target="_blank" rel="noopener">小宇宙原链接</a>' if source_url else ""
    bench_button = f'<a class="btn" href="{esc(bench_name)}" download>Benchmark JSON</a>' if bench_name else ""
    official_context_link = '<a class="btn" href="episode_page_context.md" download>官方上下文</a>' if (ep.output_dir / "episode_page_context.md").exists() else ""
    description_html = f'<p>{esc(description)}</p>' if description else '<p class="dim">暂无官方简介；重新导入小宇宙链接后会自动抓取。</p>'
    outline_html = render_outline_html(outline)
    tldr_image_name = "tldr_infographic.png" if (ep.output_dir / "tldr_infographic.png").exists() else ""
    llm_html = render_llm_summary_html(llm_summary, tldr_image_name)

    report_html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}｜播客 ASR 报告</title><style>{CSS}</style></head><body><header class="appbar"><nav class="appbar-inner"><div class="brand">Podcast ASR Studio<small>播客转写</small></div><div class="nav"><a href="../index.html">播客转写</a><a href="/static/meeting-asr/index.html">会议转写</a><a href="/legacy">旧版</a><a href="/healthz">服务状态</a></div></nav></header><main class="shell"><section class="page-header episode-hero"><h1>{esc(title)}</h1><p>{esc(llm_tldr or description[:220])}</p><div class="meta"><span>{esc(podcast_title or 'Podcast')}</span><span>{esc(trans.get('duration_formatted') or fmt_ts(duration_seconds))}</span><span>{esc(published_time or 'published time unknown')}</span><span>{len(ok_chunks)}/{len(chunks)} chunks</span></div><div class="metrics">{metrics_html}</div><div class="actions"><a class="btn primary" href="#summary">读总结</a><a class="btn" href="#outline">官方 OUTLINE</a><a class="btn" href="full.html">打开全文</a>{source_link}</div></section><div class="episode-layout"><aside class="panel toc" aria-label="报告章节"><a href="#official">官方介绍</a><a href="#outline">官方 OUTLINE</a><a href="#summary">LLM 总结</a><a href="#metrics">运行指标</a><a href="#downloads">下载与全文</a></aside><div class="content"><section class="panel" id="official"><h2>官方介绍 / Show Notes</h2>{description_html}</section><section class="panel" id="outline"><h2>官方 OUTLINE</h2>{outline_html}</section><section class="panel" id="summary"><h2>LLM 总结</h2><p class="dim">模型：{esc(llm_summary.get('summary_model') or '')} · 生成时间：{esc(llm_summary.get('generated_at') or '')} · 页面上下文：{esc('已使用' if llm_summary.get('has_page_context') or page_ctx else '未使用')}</p><div class="downloads"><a class="btn primary" href="podcast_summary.md" download>下载总结 Markdown</a><a class="btn" href="podcast_summary.json" download>下载总结 JSON</a></div>{llm_html}</section><section class="panel" id="metrics"><h2>运行指标 / CPU / GPU 对比</h2><div class="compare"><div><h3>完整转写</h3><p>{len(ok_chunks)}/{len(chunks)} · 覆盖 {fmt_ts(merged[0][0]) if merged else '—'} → {fmt_ts(merged[-1][1]) if merged else '—'}</p></div><div><h3>Benchmark</h3><p>{float(speedups.get('cuda_vs_cpu_wall') or 0):.2f}× wall · 同一 {float(bench.get('audio_seconds') or 0):.0f}s 片段</p></div></div><div class="table-wrap"><table><thead><tr><th>设备</th><th>Model load</th><th>Inference</th><th>Wall</th><th>RTF</th><th>字符</th></tr></thead><tbody><tr><td>CPU</td><td>{float(cpu.get('model_load_seconds') or 0):.2f}s</td><td>{float(cpu.get('inference_seconds') or 0):.2f}s</td><td>{float(cpu.get('wall_seconds') or 0):.2f}s</td><td>{float(cpu.get('rtf_inference') or 0):.4f}</td><td>{int(cpu.get('chars') or 0):,}</td></tr><tr><td>GPU / CUDA</td><td>{float(gpu.get('model_load_seconds') or 0):.2f}s</td><td>{float(gpu.get('inference_seconds') or 0):.2f}s</td><td>{float(gpu.get('wall_seconds') or 0):.2f}s</td><td>{float(gpu.get('rtf_inference') or 0):.4f}</td><td>{int(gpu.get('chars') or 0):,}</td></tr></tbody></table></div></section><section class="panel" id="downloads"><h2>完整转写稿与下载</h2><div class="downloads"><a class="btn primary" href="full.html">全文网页</a><a class="btn" href="{esc(transcript_md)}" download>Markdown 全文</a><a class="btn" href="{esc(transcript_txt)}" download>TXT 全文</a><a class="btn" href="{esc(transcript_srt)}" download>SRT 字幕</a><a class="btn" href="{esc(transcription_json)}" download>JSON</a>{bench_button}{official_context_link}<a class="btn" href="asr_results.zip" download>结果包 ZIP</a></div><h3>分段性能明细</h3><div class="table-wrap"><table><thead><tr><th>#</th><th>时间</th><th>音频长</th><th>字符</th><th>GPU inference</th><th>RTF</th><th>状态</th></tr></thead><tbody>{''.join(chunk_rows)}</tbody></table></div></section></div></div><footer class="footer">Auto Podcast ASR Publisher · {esc(published_at)}</footer></main></body></html>'''

    full_html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}｜完整转写全文</title><style>{CSS}</style></head><body><header class="appbar"><nav class="appbar-inner"><a class="btn" href="index.html">返回报告</a><div class="brand">Podcast ASR Studio<small>完整转写</small></div></nav></header><main class="shell"><section class="page-header"><h1>完整转写全文</h1><p>{esc(title)}</p><div class="actions">{source_link}<a class="btn" href="{esc(transcript_md)}" download>Markdown</a><a class="btn" href="{esc(transcript_txt)}" download>TXT</a></div></section><section><div class="toolbar transcript-toolbar"><input id="searchBox" class="search" placeholder="搜索全文"><button class="btn" id="clearSearch">清除</button><span class="live-count" id="matchCount">{len(chunks)} 段</span></div><div id="chunks">{''.join(transcript_sections)}</div></section><footer class="footer">全文网页链接：<code>full.html</code></footer></main><script>{SEARCH_JS}</script></body></html>'''

    episode_json = {
        "slug": ep.slug,
        "title": title,
        "podcast": podcast_title,
        "published_time": published_time,
        "description_snippet": re.sub(r"\s+", " ", description).strip()[:260],
        "official_outline_count": len(outline),
        "episode_id": trans.get("episode_id"),
        "source_url": source_url,
        "report_path": f"{ep.slug}/index.html",
        "full_text_path": f"{ep.slug}/full.html",
        "markdown_path": f"{ep.slug}/{transcript_md}",
        "txt_path": f"{ep.slug}/{transcript_txt}",
        "summary_markdown_path": f"{ep.slug}/podcast_summary.md" if (ep.output_dir / "podcast_summary.md").exists() else "",
        "summary_json_path": f"{ep.slug}/podcast_summary.json" if (ep.output_dir / "podcast_summary.json").exists() else "",
        "tldr_infographic_path": f"{ep.slug}/tldr_infographic.png" if (ep.output_dir / "tldr_infographic.png").exists() else "",
        "page_context_path": f"{ep.slug}/episode_page_context.md" if (ep.output_dir / "episode_page_context.md").exists() else "",
        "llm_tldr": llm_tldr,
        "summary_model": llm_summary.get("summary_model") or "",
        "duration_seconds": duration_seconds,
        "duration_formatted": trans.get("duration_formatted") or fmt_ts(duration_seconds),
        "chars": int(summary.get("chars") or sum(int(c.get("chars") or 0) for c in chunks)),
        "chunks": len(chunks),
        "ok_chunks": len(ok_chunks),
        "failed_chunks": len(failed_chunks),
        "coverage_pct": round(coverage, 3),
        "gpu_pipeline_wall_seconds": summary.get("pipeline_wall_seconds") or summary.get("wall_seconds"),
        "gpu_x_realtime_wall": summary.get("x_realtime_wall"),
        "gpu_vs_cpu_inference": speedups.get("cuda_vs_cpu_inference"),
        "gpu_vs_cpu_wall": speedups.get("cuda_vs_cpu_wall"),
        "published_at": published_at,
        "signature": ep.signature,
    }
    return report_html, full_html, episode_json


def publish_episode(ep: Episode) -> dict[str, Any]:
    report_html, full_html, episode_json = render_episode(ep)
    for dest in [ep.dest_dir] + ([ep.legacy_dir] if ep.legacy_dir.exists() and ep.legacy_dir != ep.dest_dir else []):
        copy_artifacts(ep, dest)
        (dest / "index.html").write_text(report_html, encoding="utf-8")
        (dest / "full.html").write_text(full_html, encoding="utf-8")
        write_json(dest / "episode.json", episode_json)
    return episode_json


def render_site_index(episodes: list[dict[str, Any]]) -> str:
    episodes_sorted = sorted(
        episodes,
        key=lambda x: (x.get("published_time") or "", x.get("published_at") or ""),
        reverse=True,
    )
    total_seconds = sum(float(ep.get("duration_seconds") or 0) for ep in episodes_sorted)
    total_chars = sum(int(ep.get("chars") or 0) for ep in episodes_sorted)
    total_chunks = sum(int(ep.get("chunks") or 0) for ep in episodes_sorted)
    ok_chunks = sum(int(ep.get("ok_chunks") or 0) for ep in episodes_sorted)
    avg_realtime_values = [float(ep.get("gpu_x_realtime_wall") or 0) for ep in episodes_sorted if ep.get("gpu_x_realtime_wall")]
    avg_realtime = sum(avg_realtime_values) / len(avg_realtime_values) if avg_realtime_values else 0
    cards = []
    for ep in episodes_sorted:
        source = ep.get("source_url") or ""
        searchable = " ".join(str(ep.get(k) or "") for k in ["title", "podcast", "llm_tldr", "description_snippet"])
        cards.append(f'''<article class="card episode" data-text="{esc(searchable)}"><div class="meta"><span>{esc(ep.get('duration_formatted'))}</span><span>{int(ep.get('chars') or 0):,} 字</span><span>{ep.get('ok_chunks')}/{ep.get('chunks')} chunks</span><span>OUTLINE {int(ep.get('official_outline_count') or 0)}</span></div><h3><a href="{esc(ep.get('report_path'))}">{esc(ep.get('title'))}</a></h3><p class="summary">{esc(ep.get('llm_tldr') or ep.get('description_snippet') or '')}</p><div class="metric-line"><span>ASR {float(ep.get('gpu_x_realtime_wall') or 0):.1f}×</span><span>覆盖率 {float(ep.get('coverage_pct') or 0):.0f}%</span><span>CPU/GPU {float(ep.get('gpu_vs_cpu_inference') or 0):.2f}×</span><span>已发布</span></div><div class="actions"><a class="btn primary" href="{esc(ep.get('report_path'))}">报告</a><a class="btn" href="{esc(ep.get('full_text_path'))}">全文</a><a class="btn" href="{esc(ep.get('summary_markdown_path') or ep.get('report_path'))}">总结</a>{f'<a class="btn" href="{esc(source)}" target="_blank" rel="noopener">小宇宙</a>' if source else ''}</div></article>''')
    updated = datetime.now().isoformat(timespec="seconds")
    empty_library = '<div class="empty-state">暂无已完成的播客转写。</div>' if not cards else ''
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Podcast ASR Studio</title><style>{CSS}</style></head><body><header class="appbar"><nav class="appbar-inner"><div class="brand">Podcast ASR Studio<small>播客转写</small></div><div class="nav"><a href="#import" aria-current="page">播客转写</a><a href="/static/meeting-asr/index.html">会议转写</a><a href="/legacy">旧版</a><a href="/healthz">服务状态</a></div></nav></header><main class="wrap"><section class="page-header"><h1>播客 ASR 索引</h1><p>贴入小宇宙链接，后台会生成可检索的转写报告、总结和下载文件。</p></section><section class="panel import" id="import"><div class="section-head"><h2>导入播客</h2><span class="section-label">URL / 音频文件</span></div><div class="tabs" role="tablist" aria-label="导入方式"><button class="tab active" type="button" role="tab" aria-selected="true" aria-controls="urlPane" data-tab="url">贴入链接</button><button class="tab" type="button" role="tab" aria-selected="false" aria-controls="filePane" tabindex="-1" data-tab="file">上传音频</button></div><div id="urlPane" role="tabpanel"><div class="field"><input id="episodeUrl" aria-label="小宇宙 episode 链接" placeholder="https://www.xiaoyuzhoufm.com/episode/..." /><button class="btn primary" id="mockRun">创建任务</button></div><p class="dim">会抓取官方标题、简介、OUTLINE、封面和音频 URL，再进入 ASR / LLM 总结。</p><div id="taskStatus" class="task-status" role="status" aria-live="polite" style="display:none"></div></div><div id="filePane" role="tabpanel" hidden><div class="drop">会议录音支持 MP3 / WAV，并自动区分发言人。<br><a class="btn primary" href="/static/meeting-asr/index.html">打开会议转写</a></div></div><div class="pipeline"><div class="step"><strong>1. 抓取介绍</strong>标题 / show notes / outline</div><div class="step"><strong>2. 远程 ASR</strong>x570 SenseVoice 转写</div><div class="step"><strong>3. LLM 总结</strong>Qwen 结合官方信息</div><div class="step"><strong>4. 发布</strong>索引 + 单集页面</div></div></section><section class="stats"><div class="stat"><small>已收录</small><strong>{len(episodes_sorted)}</strong><span class="dim">episodes</span></div><div class="stat"><small>总音频</small><strong>{fmt_ts(total_seconds)}</strong><span class="dim">已转写</span></div><div class="stat"><small>ASR 成功率</small><strong>{(ok_chunks / total_chunks * 100 if total_chunks else 0):.0f}%</strong><span class="dim">{ok_chunks} / {total_chunks} chunks</span></div><div class="stat"><small>平均 ASR 倍速</small><strong>{avg_realtime:.0f}×</strong><span class="dim">wall realtime</span></div></section><section id="library"><div class="section-head library-head"><h2>已转写播客索引</h2><div class="library-toolbar"><input class="search" id="filter" placeholder="搜索标题 / 摘要 / SpaceX / Coding" /><span class="live-count"><span id="shownCount">{len(cards)}</span> / {len(cards)} 集</span></div></div>{empty_library}<div class="grid" id="episodes">{''.join(cards)}</div><div class="empty-state" id="noResults" hidden>没有匹配的播客。<button class="btn" type="button" id="clearFilter">清除搜索</button></div></section><footer class="footer">Updated {esc(updated)} · publisher: publish_podcast_asr_site.py</footer></main><div class="toast" id="toast" role="status" aria-live="polite"></div><script>{INDEX_JS}</script></body></html>'''


def publish_all(only_if_changed: bool = False) -> dict[str, Any]:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LIBRARY_DIR / ".publish_state.json"
    old_state = read_json(state_path, {}) or {}
    old_sigs = old_state.get("signatures") or {}
    episodes = discover_episodes()
    published: list[dict[str, Any]] = []
    changed_slugs: list[str] = []
    existing_episode_jsons: dict[str, dict[str, Any]] = {}
    for ep_json_path in LIBRARY_DIR.glob("*/episode.json"):
        data = read_json(ep_json_path, {}) or {}
        if data.get("slug"):
            existing_episode_jsons[data["slug"]] = data

    for ep in episodes:
        must_publish = (
            old_sigs.get(ep.slug) != ep.signature
            or not (ep.dest_dir / "index.html").exists()
            or not (ep.dest_dir / "full.html").exists()
        )
        if must_publish or not only_if_changed:
            ep_json = publish_episode(ep)
            changed_slugs.append(ep.slug) if must_publish else None
        else:
            ep_json = read_json(ep.dest_dir / "episode.json", {}) or existing_episode_jsons.get(ep.slug, {})
        if ep_json:
            published.append(ep_json)

    index_missing = not (LIBRARY_DIR / "index.html").exists()
    if published and (changed_slugs or index_missing or not only_if_changed):
        episodes_sorted = sorted(
            published,
            key=lambda x: (x.get("published_time") or "", x.get("published_at") or ""),
            reverse=True,
        )
        write_json(LIBRARY_DIR / "episodes.json", episodes_sorted)
        index_html = render_site_index(episodes_sorted)
        index_html = index_html.replace(
            '<a class="btn" href="/legacy">旧版上传工具</a>',
            '<a class="btn" href="/static/meeting-asr/index.html">会议转写</a>'
            '<a class="btn" href="/legacy">旧版上传工具</a>',
        )
        index_html = index_html.replace(
            '<div class="drop">拖入 M4A / MP3 / WAV，或点击选择文件。<br>'
            '<span class="dim">当前为静态入口；后续接入后台任务 API。</span></div>',
            '<div class="drop">会议录音支持 MP3 / WAV，并自动区分发言人。<br>'
            '<a class="btn primary" style="margin-top:12px" href="/static/meeting-asr/index.html">打开会议转写</a></div>',
        )
        index_html = index_html.replace(
            '<div class="step"><strong>2. GPU ASR</strong>SenseVoice 分片转写</div>',
            '<div class="step"><strong>2. 远程 ASR</strong>x570 SenseVoice 转写</div>',
        )
        (LIBRARY_DIR / "index.html").write_text(index_html, encoding="utf-8")

    new_sigs = {ep.slug: ep.signature for ep in episodes}
    write_json(state_path, {"updated_at": datetime.now().isoformat(timespec="seconds"), "signatures": new_sigs})
    lan = get_lan_ip()
    return {
        "library_path": str(LIBRARY_DIR / "index.html"),
        "library_url_local": f"http://127.0.0.1:{DEFAULT_PORT}/static/{LIBRARY_SLUG}/index.html",
        "library_url_lan": f"http://{lan}:{DEFAULT_PORT}/static/{LIBRARY_SLUG}/index.html",
        "episodes": len(published),
        "changed_slugs": changed_slugs,
        "episode_urls": [f"http://{lan}:{DEFAULT_PORT}/static/{LIBRARY_SLUG}/{ep['slug']}/index.html" for ep in published],
        "full_text_urls": [f"http://{lan}:{DEFAULT_PORT}/static/{LIBRARY_SLUG}/{ep['slug']}/full.html" for ep in published],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-if-changed", action="store_true", help="Print nothing and avoid rewrite when no completed ASR output changed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output unless --only-if-changed finds changes")
    args = parser.parse_args()
    result = publish_all(only_if_changed=args.only_if_changed)
    if args.only_if_changed and not result["changed_slugs"]:
        return 0
    if not args.quiet or result["changed_slugs"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
