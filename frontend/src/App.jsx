import React, { useEffect, useRef, useState } from 'react'

// Dependency / tooling directories that must NOT be uploaded.
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

// Backend base URL. In dev leave empty and let Vite proxy /api -> :8000.
// For a separately-deployed backend, set VITE_API_BASE at build time.
const API_BASE = import.meta.env.VITE_API_BASE || ''

export default function App() {
  const inputRef = useRef(null)
  const [method, setMethod] = useState('path') // 'path' (recommended) | 'upload'
  const [path, setPath] = useState('')
  const [files, setFiles] = useState([])
  const [skipped, setSkipped] = useState(0)
  const [mode, setMode] = useState('llm')
  const [reading, setReading] = useState(false)
  const [converting, setConverting] = useState(false)
  const [status, setStatus] = useState(null) // {kind:'ok'|'err', text}

  // webkitdirectory / directory are non-standard; set them imperatively.
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.setAttribute('webkitdirectory', '')
      inputRef.current.setAttribute('directory', '')
    }
  }, [method])

  async function onPick(e) {
    const list = Array.from(e.target.files || [])
    setStatus(null)
    setReading(true)
    setFiles([])
    setSkipped(0)
    await nextFrame() // let the spinner paint before the heavy read loop

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

  function download(blob) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'converted_maf_agent.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  async function onConvert() {
    setConverting(true)
    setStatus(null)
    try {
      const endpoint = method === 'path' ? '/api/convert-path' : '/api/convert'
      const body = method === 'path' ? { mode, path } : { mode, files }
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Conversion failed' }))
        throw new Error(err.detail || 'Conversion failed')
      }
      download(await res.blob())
      setStatus({ kind: 'ok', text: 'Done — converted_maf_agent.zip downloaded. See INSTALL.md & MIGRATION_REPORT.md inside.' })
    } catch (e) {
      // A TypeError from fetch means the request never reached the backend.
      const networkError = e instanceof TypeError || /failed to fetch/i.test(e.message)
      setStatus({
        kind: 'err',
        text: networkError
          ? 'Cannot reach the backend. Start it with: python -m uvicorn api:app --app-dir backend --port 8000, then open this page at http://localhost:8000'
          : e.message,
      })
    } finally {
      setConverting(false)
    }
  }

  const canConvert = method === 'path' ? path.trim().length > 0 : files.length > 0

  return (
    <div className="wrap">
      <h1>Framework Conversion Utility</h1>
      <p className="sub">
        Convert a <b>LangGraph</b> agent to the <b>Microsoft Agent Framework (MAF)</b>.
        Point at the agent folder, choose how the hard parts are handled, and download the converted agent.
      </p>

      <div className="card">
        <h2>Choose the agent folder</h2>
        <div className="opts">
          <div className={'opt' + (method === 'path' ? ' active' : '')} onClick={() => setMethod('path')}>
            <h3>Local folder path (recommended)</h3>
            <p>The backend reads the folder from disk. Nothing is uploaded, and dependency dirs (.venv, node_modules, …) are ignored automatically — best for large folders.</p>
          </div>
          <div className={'opt' + (method === 'upload' ? ' active' : '')} onClick={() => setMethod('upload')}>
            <h3>Upload folder</h3>
            <p>Use when the backend is on a different machine. The browser stages every file, so avoid folders that contain a virtualenv.</p>
          </div>
        </div>

        {method === 'path' ? (
          <div style={{ marginTop: 14 }}>
            <input
              type="text"
              className="path"
              placeholder="e.g. C:\Users\you\my-agent   (folder that has README.md + .py)"
              value={path}
              onChange={(e) => { setPath(e.target.value); setStatus(null) }}
            />
          </div>
        ) : (
          <label className="file" style={{ marginTop: 14 }}>
            Click to choose your agent folder
            <input ref={inputRef} type="file" multiple onChange={onPick} />
            <div className="picked">
              {reading && <span className="spinner" />}
              {reading
                ? <span>uploading…</span>
                : files.length
                  ? <span>{files.length} source files selected ({skipped} dependency/other files skipped)</span>
                  : <span>No folder selected yet</span>}
            </div>
          </label>
        )}
      </div>

      <div className="card">
        <h2>How should the hard parts (HITL, complex orchestration) be handled?</h2>
        <div className="opts">
          <div className={'opt' + (mode === 'llm' ? ' active' : '')} onClick={() => setMode('llm')}>
            <h3>LLM does it, I review</h3>
            <p>Gemini writes the approval flow &amp; complex orchestration. Kept as a reviewable commented block; flagged in the migration report.</p>
          </div>
          <div className={'opt' + (mode === 'manual' ? ' active' : '')} onClick={() => setMode('manual')}>
            <h3>I'll do it manually</h3>
            <p>Hard parts are left as clearly-marked stubs/commented templates for you to implement. No LLM is used.</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Convert</h2>
        <button onClick={onConvert} disabled={!canConvert || reading || converting}>
          {converting && <span className="spinner dark" />}
          {converting ? 'Converting…' : 'Convert & download'}
        </button>
        {status && <div className={'status ' + status.kind}>{status.text}</div>}
      </div>
    </div>
  )
}
