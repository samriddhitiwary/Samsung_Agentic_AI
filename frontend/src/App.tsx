import { Activity, BarChart3, Boxes, Clock3, GitBranch, History, Play, Search, Server, Zap } from "lucide-react";
import { KeyboardEvent, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { getJson, postJson, QueryResponse, QueryResult } from "./lib/api";

type Workspace = "search" | "index" | "versions" | "update" | "history" | "evaluation";

const frozenMetrics = {
  p0: [
    ["NDCG@10", "0.86950"],
    ["MRR@10", "0.84141"],
    ["HitRate@10", "95.564%"],
    ["Recall@100", "99.097%"]
  ],
  p1: [
    ["Avg embedding reuse", "86.83%"],
    ["Embedding work saved", "96.42%"],
    ["Max measured speedup", "15.54×"],
    ["Retrieval parity", "PASS"]
  ],
  evo: [
    ["Grouped HitRate@5", "0.6667"],
    ["Grouped MRR", "0.3667"],
    ["Duplicate top-5", "15 → 0"]
  ]
};

const examples = [
  "Where is openBluetoothSettings used?",
  "Which functions call validate before save?",
  "How is a signature created?",
  "Where is SHA1 signer used before serialization?",
  "Where is the Bluetooth-settings deeplink used?"
];

export function App() {
  const [workspace, setWorkspace] = useState<Workspace>("search");
  const [repoId, setRepoId] = useState("itsdangerous_api_demo");
  const [commit, setCommit] = useState("");
  const [query, setQuery] = useState("How is a signature created?");
  const [topK, setTopK] = useState(10);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [versionA, setVersionA] = useState("");
  const [versionB, setVersionB] = useState("");
  const [versionLeft, setVersionLeft] = useState<QueryResponse | null>(null);
  const [versionRight, setVersionRight] = useState<QueryResponse | null>(null);
  const [registerPath, setRegisterPath] = useState("");
  const [registerRef, setRegisterRef] = useState("HEAD");
  const [registerResult, setRegisterResult] = useState<Record<string, unknown> | null>(null);
  const [updateTarget, setUpdateTarget] = useState("");
  const [updateResult, setUpdateResult] = useState<Record<string, unknown> | null>(null);
  const [historyResult, setHistoryResult] = useState<Record<string, unknown> | null>(null);
  const [recent, setRecent] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getJson<Record<string, unknown>>("/health").then(setHealth).catch((err) => setError(`Backend offline: ${err.message}`));
  }, []);

  async function runQuery(targetCommit = commit) {
    setBusy(true);
    setError(null);
    const started = performance.now();
    try {
      const payload = await postJson<QueryResponse>("/query", {
        repo_id: repoId,
        commit: targetCommit || undefined,
        query,
        top_k: topK
      });
      payload.timing = { ...payload.timing, browser_roundtrip_ms: performance.now() - started };
      setResponse(payload);
      setRecent((items) => [query, ...items.filter((item) => item !== query)].slice(0, 8));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function registerRepo() {
    setBusy(true);
    setError(null);
    try {
      const payload = await postJson<Record<string, unknown>>("/repos/register", {
        repo_path: registerPath,
        repo_id: repoId || undefined,
        commit: registerRef || "HEAD"
      });
      setRegisterResult(payload);
      setRepoId(String(payload.repo ?? repoId));
      setCommit(String(payload.commit ?? ""));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function updateRepo() {
    setBusy(true);
    setError(null);
    try {
      const payload = await postJson<Record<string, unknown>>(`/repos/${repoId}/update`, { commit: updateTarget });
      setUpdateResult(payload);
      setCommit(String(payload.new_commit ?? ""));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function compareVersions() {
    setBusy(true);
    setError(null);
    try {
      const [left, right] = await Promise.all([
        postJson<QueryResponse>("/query", { repo_id: repoId, commit: versionA || undefined, query, top_k: topK }),
        postJson<QueryResponse>("/query", { repo_id: repoId, commit: versionB || undefined, query, top_k: topK })
      ]);
      setVersionLeft(left);
      setVersionRight(right);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function searchHistory() {
    setBusy(true);
    setError(null);
    try {
      setHistoryResult(await postJson<Record<string, unknown>>("/search/evolution", {
        repo_id: repoId,
        query,
        top_k: topK,
        include_evolution_context: true
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.ctrlKey && event.key === "Enter") runQuery();
  }

  const nav = useMemo(() => [
    ["search", Search, "Search Code"],
    ["index", Boxes, "Index Repository"],
    ["versions", GitBranch, "Versions"],
    ["update", Zap, "Update Index"],
    ["history", History, "Search Across History"],
    ["evaluation", BarChart3, "Evaluation"]
  ] as const, []);

  return (
    <div className="min-h-screen bg-graphite text-slate-100">
      <aside className="fixed inset-y-0 left-0 w-72 border-r border-line bg-[#0b1018]/95 p-5">
        <div className="mb-8">
          <div className="text-xs uppercase tracking-[0.32em] text-samsung">Samsung Theme 1</div>
          <h1 className="mt-3 text-2xl font-semibold">Agentic Code Intelligence</h1>
          <p className="mt-2 text-sm text-slate-400">Search and understand large codebases using natural language.</p>
        </div>
        <nav className="space-y-2">
          {nav.map(([id, Icon, label]) => (
            <button key={id} onClick={() => setWorkspace(id)} className={`nav-item ${workspace === id ? "nav-active" : ""}`}>
              <Icon size={17} /> {label}
            </button>
          ))}
        </nav>
        <div className="absolute bottom-5 left-5 right-5 rounded-xl border border-line bg-panel p-3 text-xs text-slate-400">
          <div className="mb-2 flex items-center gap-2 text-slate-200"><Server size={14} /> System</div>
          <div>Backend: {health ? "online" : "checking"}</div>
          <div>Embedding: {String((health?.llama_cpp as Record<string, unknown> | undefined)?.server_healthy ?? "unknown")}</div>
        </div>
      </aside>

      <main className="ml-72 p-6">
        {error && <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-100">{error}</div>}
        {workspace === "search" && (
          <Panel title="Search Code" icon={<Search size={19} />}>
            <SearchForm repoId={repoId} setRepoId={setRepoId} commit={commit} setCommit={setCommit} query={query} setQuery={setQuery} topK={topK} setTopK={setTopK} onSearch={() => runQuery()} onKeyDown={onKeyDown} busy={busy} />
            <ExampleQueries setQuery={setQuery} />
            <TraceAndResults response={response} />
            <HistoryBox recent={recent} setQuery={setQuery} />
          </Panel>
        )}

        {workspace === "index" && (
          <Panel title="Index Repository" icon={<Boxes size={19} />}>
            <div className="grid gap-3 md:grid-cols-3">
              <Input label="Repository ID" value={repoId} onChange={setRepoId} />
              <Input label="Local repository path" value={registerPath} onChange={setRegisterPath} placeholder="C:\\path\\to\\repo" />
              <Input label="Git ref / commit" value={registerRef} onChange={setRegisterRef} />
            </div>
            <button className="primary mt-4" disabled={busy || !registerPath} onClick={registerRepo}>Index Repository</button>
            <ObjectPanel title="Indexing result" value={registerResult} />
          </Panel>
        )}

        {workspace === "versions" && (
          <Panel title="Version Retrieval" icon={<GitBranch size={19} />}>
            <SearchForm repoId={repoId} setRepoId={setRepoId} commit={commit} setCommit={setCommit} query={query} setQuery={setQuery} topK={topK} setTopK={setTopK} onSearch={compareVersions} onKeyDown={onKeyDown} busy={busy} />
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <Input label="Commit A" value={versionA} onChange={setVersionA} />
              <Input label="Commit B" value={versionB} onChange={setVersionB} />
            </div>
            <button className="primary mt-4" disabled={busy} onClick={compareVersions}>Compare A ↔ B</button>
            <div className="mt-5 grid gap-4 xl:grid-cols-2">
              <ResultColumn title="Commit A" response={versionLeft} />
              <ResultColumn title="Commit B" response={versionRight} />
            </div>
          </Panel>
        )}

        {workspace === "update" && (
          <Panel title="P1 Incremental Update" icon={<Zap size={19} />}>
            <div className="grid gap-3 md:grid-cols-3">
              <Input label="Repository ID" value={repoId} onChange={setRepoId} />
              <Input label="Current commit" value={commit} onChange={setCommit} />
              <Input label="Target commit" value={updateTarget} onChange={setUpdateTarget} />
            </div>
            <button className="primary mt-4" disabled={busy || !updateTarget} onClick={updateRepo}>Update Index</button>
            <ObjectPanel title="Update result" value={updateResult} />
          </Panel>
        )}

        {workspace === "history" && (
          <Panel title="Search Across History" icon={<History size={19} />}>
            <SearchForm repoId={repoId} setRepoId={setRepoId} commit={commit} setCommit={setCommit} query={query} setQuery={setQuery} topK={topK} setTopK={setTopK} onSearch={searchHistory} onKeyDown={onKeyDown} busy={busy} />
            <ObjectPanel title="Grouped semantic states" value={historyResult} />
          </Panel>
        )}

        {workspace === "evaluation" && (
          <Panel title="Evaluation Evidence" icon={<BarChart3 size={19} />}>
            <MetricSection title="Official Samsung P0" subtitle="Official MTEB AppsRetrieval test split" rows={frozenMetrics.p0} />
            <MetricSection title="P1 Version-Aware Retrieval" subtitle="Real-repo internal benchmark" rows={frozenMetrics.p1} />
            <MetricSection title="Evolutionary Retrieval" subtitle="Internal validation, not official Samsung metric" rows={frozenMetrics.evo} />
          </Panel>
        )}
      </main>
    </div>
  );
}

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return <section className="rounded-2xl border border-line bg-[#0d131d] p-5 shadow-2xl shadow-black/20"><h2 className="mb-4 flex items-center gap-2 text-xl font-semibold">{icon}{title}</h2>{children}</section>;
}

function SearchForm(props: { repoId: string; setRepoId: (v: string) => void; commit: string; setCommit: (v: string) => void; query: string; setQuery: (v: string) => void; topK: number; setTopK: (v: number) => void; onSearch: () => void; onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void; busy: boolean }) {
  return <div className="rounded-xl border border-line bg-panel p-4">
    <div className="grid gap-3 md:grid-cols-[1fr_1fr_120px]">
      <Input label="Repository" value={props.repoId} onChange={props.setRepoId} />
      <Input label="Version / commit" value={props.commit} onChange={props.setCommit} placeholder="active commit" />
      <label className="block text-sm text-slate-300">Top K<select className="input mt-1" value={props.topK} onChange={(e) => props.setTopK(Number(e.target.value))}><option>5</option><option>10</option><option>20</option></select></label>
    </div>
    <label className="mt-4 block text-sm text-slate-300">Query</label>
    <textarea className="input mt-1 h-28 resize-none" value={props.query} onChange={(e) => props.setQuery(e.target.value)} onKeyDown={props.onKeyDown} placeholder="Ask a question about this codebase..." />
    <button className="primary mt-4" disabled={props.busy || !props.query.trim()} onClick={props.onSearch}><Play size={16} /> {props.busy ? "Running..." : "Search Code"}</button>
  </div>;
}

function Input({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return <label className="block text-sm text-slate-300">{label}<input className="input mt-1" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} /></label>;
}

function ExampleQueries({ setQuery }: { setQuery: (v: string) => void }) {
  return <div className="mt-4 flex flex-wrap gap-2">{examples.map((item) => <button key={item} className="chip" onClick={() => setQuery(item)}>{item}</button>)}</div>;
}

function TraceAndResults({ response }: { response: QueryResponse | null }) {
  if (!response) return null;
  const total = response.timing.total_ms ?? response.timing.browser_roundtrip_ms;
  return <div className="mt-5 grid gap-4 xl:grid-cols-[320px_1fr]">
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="mb-3 flex items-center gap-2 font-medium"><Activity size={16} /> Retrieval Plan</div>
      <div className="space-y-2 text-sm">{response.trace.map((step, idx) => <div key={idx} className="rounded-lg bg-black/20 p-2">✓ {String(step.stage)} {step.result ? `→ ${String(step.result)}` : ""} {step.candidates ? `→ ${String(step.candidates)} candidates` : ""} {step.matches ? `→ ${String(step.matches)} matches` : ""}</div>)}</div>
      <div className="mt-4 rounded-lg border border-line p-3 text-sm"><Clock3 size={14} className="mb-1 inline" /> Retrieved {response.results.length} snippets in {Math.round(Number(total ?? 0))} ms</div>
      <pre className="mt-3 max-h-52 overflow-auto rounded-lg bg-black/30 p-3 text-xs text-slate-300">{JSON.stringify(response.timing, null, 2)}</pre>
    </div>
    <ResultColumn title={`${response.query_type.toUpperCase()} results`} response={response} />
  </div>;
}

function ResultColumn({ title, response }: { title: string; response: QueryResponse | null }) {
  return <div className="rounded-xl border border-line bg-panel p-4"><h3 className="mb-3 font-medium">{title}</h3>{response?.results?.length ? <div className="space-y-3">{response.results.map((result) => <ResultCard key={`${result.rank}-${result.path}-${result.start_line}`} result={result} />)}</div> : <div className="text-sm text-slate-500">No results yet.</div>}</div>;
}

function ResultCard({ result }: { result: QueryResult }) {
  const calls = result.ordered_calls?.map((call) => call.name ?? call.raw).filter(Boolean);
  return <article className="rounded-xl border border-line bg-[#0a0f17] p-3">
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="rank">#{result.rank}</span>
      <span className="font-medium text-slate-100">{result.symbol ?? "anonymous"}</span>
      <span className="badge">{result.evidence_type ?? "semantic"}</span>
      <span className="ml-auto text-slate-400">{Number(result.score).toFixed(4)}</span>
    </div>
    <div className="mt-2 font-mono text-xs text-samsung">{result.path}:{result.start_line ?? "?"}-{result.end_line ?? "?"}</div>
    <div className="mt-1 text-xs text-slate-500">{result.symbol_type} · {shortSha(result.commit)}</div>
    {calls?.length ? <div className="mt-3 rounded-lg border border-blue-400/20 bg-blue-400/5 p-2 text-xs text-blue-100">{calls.join(" → ")}<div className="mt-1 text-slate-400">{result.limitation}</div></div> : null}
    <pre className="code mt-3">{withLineNumbers(result.snippet ?? "", result.start_line ?? 1)}</pre>
  </article>;
}

function MetricSection({ title, subtitle, rows }: { title: string; subtitle: string; rows: string[][] }) {
  return <div className="mb-5 rounded-xl border border-line bg-panel p-4"><h3 className="font-medium">{title}</h3><p className="text-sm text-slate-400">{subtitle}</p><div className="mt-3 grid gap-3 md:grid-cols-4">{rows.map(([k, v]) => <div key={k} className="rounded-lg bg-black/20 p-3"><div className="text-xs text-slate-500">{k}</div><div className="text-lg font-semibold">{v}</div></div>)}</div></div>;
}

function ObjectPanel({ title, value }: { title: string; value: Record<string, unknown> | null }) {
  return <div className="mt-4 rounded-xl border border-line bg-panel p-4"><h3 className="mb-2 font-medium">{title}</h3><pre className="max-h-96 overflow-auto rounded-lg bg-black/30 p-3 text-xs text-slate-300">{value ? JSON.stringify(value, null, 2) : "No data yet."}</pre></div>;
}

function HistoryBox({ recent, setQuery }: { recent: string[]; setQuery: (v: string) => void }) {
  if (!recent.length) return null;
  return <div className="mt-4 rounded-xl border border-line bg-panel p-4"><h3 className="mb-2 text-sm font-medium">Session query history</h3><div className="flex flex-wrap gap-2">{recent.map((item) => <button className="chip" key={item} onClick={() => setQuery(item)}>{item}</button>)}</div></div>;
}

function withLineNumbers(snippet: string, start: number) {
  return snippet.split("\n").map((line, idx) => `${String(start + idx).padStart(4, " ")} │ ${line}`).join("\n");
}

function shortSha(value?: string) {
  return value ? value.slice(0, 8) : "active";
}
