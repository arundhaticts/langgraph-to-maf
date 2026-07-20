import React, { useEffect, useRef, useState } from 'react'

const IGNORE_DIRS = new Set([
  '.venv', 'venv', 'env', '.env', 'node_modules', '__pycache__', '.git',
  '.hg', '.svn', 'site-packages', '.mypy_cache', '.pytest_cache', '.tox',
  '.idea', '.vscode', 'dist', 'build', '.eggs', '.cache',
])
const IGNORE_EXT = /\.(pyc|pyo|so|dll|dylib|bin|exe|zip|tar|gz|whl|png|jpe?g|gif|pdf|ico|woff2?|ttf|mp4|mov)$/i
const MAX_FILE_BYTES = 1_000_000

function isSource(rel, size) {
  const parts = rel.replace(/\\/g, '/').split('/')
  if (parts.some((p) => IGNORE_DIRS.has(p) || p.endsWith('.egg-info'))) return false
  if (IGNORE_EXT.test(rel)) return false
  if (size > MAX_FILE_BYTES) return false
  return true
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()))
const API_BASE = import.meta.env.VITE_API_BASE || ''
const NEW_FW = '__new__'

// Slugify a name to letters/digits/hyphens so the zip name matches the backend
// ({input_folder_name}-{target_framework}.zip) exactly.
const slug = (text, fallback) => {
  const s = String(text || '').trim().replace(/[^A-Za-z0-9]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase()
  return s || fallback
}

const nowTime = () =>
  new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true })

const generatedOn = () => {
  const d = new Date()
  const day = String(d.getDate()).padStart(2, '0')
  const mon = d.toLocaleString('en-US', { month: 'short' })
  const time = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
  return `${day}-${mon}-${d.getFullYear()} ${time}`
}

export default function App() {
  const agentInputRef = useRef(null)
  const packInputRef = useRef(null)
  const logTimer = useRef(null)

  const [method, setMethod] = useState('path')
  const [path, setPath] = useState('')
  const [files, setFiles] = useState([])
  const [skipped, setSkipped] = useState(0)
  const [mode, setMode] = useState('llm')
  const [frameworks, setFrameworks] = useState([])
  const [source, setSource] = useState('')
  const [target, setTarget] = useState('')
  const [packFiles, setPackFiles] = useState([])
  const [packSkipped, setPackSkipped] = useState(0)
  const [reading, setReading] = useState(false)
  const [readingPack, setReadingPack] = useState(false)
  const [converting, setConverting] = useState(false)
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [log, setLog] = useState([])

  const sourceOptions = frameworks.filter((f) => f.source)
  const targetOptions = frameworks.filter((f) => f.target && f.name !== source)
  const isNewFw = target === NEW_FW

  const targetDisplay = isNewFw
    ? 'your custom framework'
    : (frameworks.find((f) => f.name === target)?.display_name || target || 'the target framework')

  useEffect(() => {
    fetch(`${API_BASE}/api/frameworks`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r)))
      .then((data) => {
        const fw = data.frameworks || []
        setFrameworks(fw)
        const firstSource = fw.find((f) => f.source)
        const firstTarget = fw.find((f) => f.target)
        if (firstSource) setSource(firstSource.name)
        if (firstTarget) setTarget(firstTarget.name)
      })
      .catch(() => setStatus({
        kind: 'err',
        text: 'Cannot reach the backend. Start it with: python -m uvicorn api:app --app-dir backend --port 8000',
      }))
  }, [])

  useEffect(() => {
    if (agentInputRef.current) {
      agentInputRef.current.setAttribute('webkitdirectory', '')
      agentInputRef.current.setAttribute('directory', '')
    }
  }, [method])

  useEffect(() => {
    if (packInputRef.current) {
      packInputRef.current.setAttribute('webkitdirectory', '')
      packInputRef.current.setAttribute('directory', '')
    }
  }, [isNewFw])

  useEffect(() => () => clearInterval(logTimer.current), [])

  async function onPick(e) {
    const list = Array.from(e.target.files || [])
    setStatus(null)
    setReading(true)
    setFiles([])
    setSkipped(0)
    await nextFrame()
    const collected = []
    let skips = 0
    for (const f of list) {
      const rel = f.webkitRelativePath || f.name
      if (!isSource(rel, f.size)) { skips++; continue }
      try { collected.push({ path: rel, content: await f.text() }) }
      catch { skips++ }
    }
    setFiles(collected)
    setSkipped(skips)
    setReading(false)
  }

  async function onPickPack(e) {
    const list = Array.from(e.target.files || [])
    setReadingPack(true)
    setPackFiles([])
    setPackSkipped(0)
    await nextFrame()
    const collected = []
    let skips = 0
    for (const f of list) {
      const rel = f.webkitRelativePath || f.name
      if (IGNORE_EXT.test(rel) || f.size > MAX_FILE_BYTES) { skips++; continue }
      try { collected.push({ path: rel, content: await f.text() }) }
      catch { skips++ }
    }
    setPackFiles(collected)
    setPackSkipped(skips)
    setReadingPack(false)
  }

  function inferFolderName() {
    if (method === 'path' && path.trim()) {
      const parts = path.trim().replace(/\\/g, '/').split('/').filter(Boolean)
      return parts[parts.length - 1] || 'agent'
    }
    if (files.length > 0) {
      const first = files[0].path.replace(/\\/g, '/').split('/')[0]
      return first || 'agent'
    }
    return 'agent'
  }

  // ── Live activity log ──────────────────────────────────────────────────────
  function pushLog(message, kind = 'done') {
    if (!message || !String(message).trim()) return
    setLog((prev) => [...prev, { time: nowTime(), message, kind }])
  }
  function markLastDone() {
    setLog((prev) => prev.map((e, i) => (i === prev.length - 1 && e.kind === 'running' ? { ...e, kind: 'done' } : e)))
  }

  function startActivityStream() {
    const stages = [
      'Upload received',
      'Analyzing source framework',
      'Mapping agent architecture',
      'Converting workflows',
      `Generating ${targetDisplay} implementation`,
      'Creating migration report',
      'Calculating readiness metrics',
      'Packaging output',
    ]
    setLog([{ time: nowTime(), message: stages[0], kind: 'done' }])
    let i = 1
    logTimer.current = setInterval(() => {
      if (i >= stages.length) { clearInterval(logTimer.current); return }
      setLog((prev) => {
        const advanced = prev.map((e, idx) =>
          idx === prev.length - 1 && e.kind === 'running' ? { ...e, kind: 'done' } : e,
        )
        return [...advanced, { time: nowTime(), message: stages[i], kind: 'running' }]
      })
      i += 1
    }, 1300)
  }

  async function onConvert() {
    setConverting(true)
    setStatus(null)
    setResult(null)
    startActivityStream()
    try {
      const actualTarget = isNewFw ? 'dynamic' : target
      const endpoint = method === 'path' ? '/api/convert-path' : '/api/convert'
      const base = method === 'path'
        ? { mode, path, source, target: actualTarget }
        : { mode, files, source, target: actualTarget }
      const body = isNewFw ? { ...base, framework_files: packFiles } : base
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Conversion failed' }))
        throw new Error(err.detail || 'Conversion failed')
      }

      const h = (name) => res.headers.get(name) || ''
      const blob = await res.blob()
      const folderName = slug(inferFolderName(), 'agent')
      const targetName = slug(isNewFw ? 'custom' : target, 'converted')
      const filename = h('X-Zip-Filename') || `${folderName}-${targetName}.zip`

      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      a.click()
      URL.revokeObjectURL(url)

      clearInterval(logTimer.current)
      markLastDone()
      pushLog('Conversion completed', 'done')

      // Resolve target display name from the header (authoritative) → frameworks list → fallback
      const targetFwName = h('X-Target-Framework') || (isNewFw ? 'custom' : target)
      const targetFwDisplay = frameworks.find((f) => f.name === targetFwName)?.display_name || targetFwName

      const loAcc = h('X-Lowest-Accuracy')
      const hiAcc = h('X-Highest-Accuracy')
      const accRange = loAcc && hiAcc ? `${loAcc}% – ${hiAcc}%` : ''

      setResult({
        filename,
        target: targetFwDisplay,
        generatedOn: generatedOn(),
        filesConverted: h('X-Files-Converted'),
        conversionTime: h('X-Conversion-Time'),
        readinessPct: h('X-Readiness-Pct'),
        accuracy: h('X-Overall-Accuracy'),
        accRange,
        remainingWork: h('X-Total-Human-Time'),
      })
      setStatus({ kind: 'ok', text: `${filename} downloaded.` })
    } catch (e) {
      clearInterval(logTimer.current)
      const networkError = e instanceof TypeError || /failed to fetch/i.test(e.message)
      const msg = networkError
        ? 'Cannot reach the backend. Start it with: python -m uvicorn api:app --app-dir backend --port 8000'
        : e.message
      pushLog(msg, 'error')
      setStatus({ kind: 'err', text: msg })
    } finally {
      setConverting(false)
    }
  }

  const hasSource = method === 'path' ? path.trim().length > 0 : files.length > 0
  const hasPack = !isNewFw || packFiles.length > 0
  const canConvert = hasSource && !!target && hasPack

  const showRightPanel = converting || log.length > 0 || result

  return (
    <>
      {/* Brand header */}
      <header className="site-header">
        <div className="site-header-inner">
          <span className="brand-title">Framework Conversion Utility</span>
        </div>
      </header>

      <div className={'page-layout' + (showRightPanel ? ' with-panel' : '')}>
        {/* LEFT — form */}
        <main className="form-col">
          <div className="page-intro">
            <h1>Convert an agent between frameworks</h1>
            <p>Pick your source folder, choose the source and target frameworks, and download a ready-to-run package.</p>
          </div>

          {/* Step 1 */}
          <div className="card">
            <div className="section-label">Step 1 — Agent folder</div>
            <div className="toggle-row">
              <div className={'toggle-btn' + (method === 'path' ? ' active' : '')} onClick={() => setMethod('path')}>
                Local path <span className="badge">Recommended</span>
              </div>
              <div className={'toggle-btn' + (method === 'upload' ? ' active' : '')} onClick={() => setMethod('upload')}>
                Upload folder
              </div>
            </div>

            {method === 'path' ? (
              <>
                <div className="helper-text">Paste the full path to the agent folder on this machine. Dependency directories (.venv, node_modules) are ignored automatically.</div>
                <input
                  type="text"
                  className="path-input"
                  placeholder="e.g. C:\Users\you\my-agent"
                  value={path}
                  onChange={(e) => { setPath(e.target.value); setStatus(null) }}
                />
              </>
            ) : (
              <>
                <div className="helper-text">Use when the backend is on a different machine. Avoid uploading folders that contain a virtualenv.</div>
                <label className="file-drop">
                  Click to select your agent folder
                  <input ref={agentInputRef} type="file" multiple onChange={onPick} />
                  <div className="picked">
                    {reading && <span className="spinner" />}
                    {reading ? <span>Reading files…</span>
                      : files.length ? <span>{files.length} files selected{skipped > 0 ? ` (${skipped} skipped)` : ''}</span>
                      : <span>No folder selected</span>}
                  </div>
                </label>
              </>
            )}
          </div>

          {/* Step 2 */}
          <div className="card">
            <div className="section-label">Step 2 — Source &amp; target framework</div>
            <div className="fw-row">
              <div className="field">
                <label>From</label>
                <div className="select-wrap">
                  <select className="path-input" value={source} onChange={(e) => { setSource(e.target.value); setStatus(null) }}>
                    {sourceOptions.length === 0 && <option value="">(loading…)</option>}
                    {sourceOptions.map((f) => (
                      <option key={f.name} value={f.name}>{f.display_name}</option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="fw-arrow">→</div>
              <div className="field">
                <label>To</label>
                <div className="select-wrap">
                  <select className="path-input" value={target} onChange={(e) => { setTarget(e.target.value); setStatus(null) }}>
                    {targetOptions.length === 0 && <option value="">(loading…)</option>}
                    {targetOptions.map((f) => (
                      <option key={f.name} value={f.name}>{f.display_name}</option>
                    ))}
                    <option value={NEW_FW}>New / custom framework…</option>
                  </select>
                </div>
              </div>
            </div>

            {isNewFw && (
              <div className="pack-upload-box">
                <div className="pack-title">Upload your framework folder</div>
                <div className="pack-hint">
                  Your framework folder must contain the following three items:
                  <ul className="pack-list">
                    <li>
                      <code>vocabulary.json</code> — <strong>Required.</strong> The machine-readable construct map that drives the converter's deterministic translation. It declares the framework's agent classes, tool decorator, context class, runtime dependencies, and a <code>"capabilities"</code> block listing which constructs are supported natively, supported with data loss, or unsupported.
                    </li>
                    <li>
                      <code>docs.md</code> — <strong>Required.</strong> A human-readable overview of the framework: design philosophy, primary classes, wiring patterns, and a short usage example. The LLM reads this to generate the complex orchestration sections.
                    </li>
                    <li>
                      <code>examples/</code> — <strong>Required.</strong> One or more working Python agent files. Used as few-shot context when generating tool wrappers and orchestration code. Aim for at least one simple and one multi-tool example.
                    </li>
                  </ul>
                </div>
                <label className="file-drop" style={{ marginTop: 10 }}>
                  Click to select your framework folder
                  <input ref={packInputRef} type="file" multiple onChange={onPickPack} />
                  <div className="picked">
                    {readingPack && <span className="spinner" />}
                    {readingPack ? <span>Reading…</span>
                      : packFiles.length ? <span>{packFiles.length} files loaded{packSkipped > 0 ? ` (${packSkipped} skipped)` : ''}</span>
                      : <span>No folder selected</span>}
                  </div>
                </label>
              </div>
            )}
          </div>

          {/* Step 3 */}
          <div className="card">
            <div className="section-label">Step 3 — How to handle complex parts</div>
            <div className="mode-row">
              <div className={'mode-opt' + (mode === 'llm' ? ' active' : '')} onClick={() => setMode('llm')}>
                <h3>LLM assists, I review</h3>
                <p>The LLM drafts approval flows and complex orchestration. Output is a reviewable commented block, flagged in the report.</p>
              </div>
              <div className={'mode-opt' + (mode === 'manual' ? ' active' : '')} onClick={() => setMode('manual')}>
                <h3>I'll complete it manually</h3>
                <p>Hard parts are left as clearly-marked stubs for you to implement. No LLM is used.</p>
              </div>
            </div>
          </div>

          {/* Step 4 */}
          <div className="card">
            <div className="section-label">Step 4 — Convert</div>
            <button className="convert-btn" onClick={onConvert} disabled={!canConvert || reading || converting}>
              {converting && <span className="spinner on-btn" />}
              {converting ? 'Converting…' : 'Convert & download'}
            </button>
            {status && <div className={'status ' + status.kind}>{status.text}</div>}
          </div>
        </main>

        {/* RIGHT — live activity + result cards */}
        {showRightPanel && (
          <aside className="results-col">
            {/* Live activity log — stays visible after completion */}
            {log.length > 0 && (
              <div className="log-card">
                <div className="log-title">Conversion Activity</div>
                <div className="log-stream">
                  {log.filter((e) => e.message && e.message.trim()).map((e, i) => (
                    <div className="log-entry" key={i}>
                      <div className="log-time">{e.time}</div>
                      <div className={'log-msg ' + e.kind}>
                        <span className="log-icon">{e.kind === 'error' ? '✕' : e.kind === 'warn' ? '!' : e.kind === 'running' ? '◌' : '✓'}</span>
                        <span className="log-text">{e.message}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Result cards */}
            {result && (
              <>
                {/* CARD 1 — Complete */}
                <div className="card kpi-card">
                  <div className="kpi-head done">✅ Conversion Complete</div>
                  <div className="kpi-list">
                    <div className="kpi-row">
                      <span className="kpi-key">Output</span>
                      <span className="kpi-val mono">{result.filename}</span>
                    </div>
                    <div className="kpi-row">
                      <span className="kpi-key">Generated On</span>
                      <span className="kpi-val">{result.generatedOn}</span>
                    </div>
                  </div>
                </div>

                {/* CARD 2 — Summary */}
                <div className="card kpi-card">
                  <div className="kpi-head">Conversion Summary</div>
                  <div className="kpi-grid">
                    <div className="kpi-cell">
                      <div className="kpi-label">Target Framework</div>
                      <div className="kpi-metric">{result.target}</div>
                    </div>
                    {result.filesConverted && (
                      <div className="kpi-cell">
                        <div className="kpi-label">Files Converted</div>
                        <div className="kpi-metric">{result.filesConverted}</div>
                      </div>
                    )}
                    {result.conversionTime && (
                      <div className="kpi-cell">
                        <div className="kpi-label">Time Taken</div>
                        <div className="kpi-metric">{result.conversionTime}</div>
                      </div>
                    )}
                    {result.readinessPct && (
                      <div className="kpi-cell">
                        <div className="kpi-label">Overall Readiness</div>
                        <div className="kpi-metric accent">{result.readinessPct}%</div>
                      </div>
                    )}
                    {result.accuracy && (
                      <div className="kpi-cell">
                        <div className="kpi-label">End-to-End Accuracy</div>
                        <div className="kpi-metric accent">{result.accuracy}</div>
                        {result.accRange && <div className="kpi-sub">{result.accRange} range</div>}
                      </div>
                    )}
                    {result.remainingWork && (
                      <div className="kpi-cell">
                        <div className="kpi-label">Remaining Work</div>
                        <div className="kpi-metric">{result.remainingWork}</div>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </aside>
        )}
      </div>
    </>
  )
}
