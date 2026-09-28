import {
  Activity,
  BarChart3,
  Boxes,
  CheckCircle2,
  Clock3,
  GitBranch,
  History,
  Loader2,
  Play,
  Search,
  Server,
  Zap
} from "lucide-react";
import { KeyboardEvent, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { getJson, postJson, QueryResponse, QueryResult } from "./lib/api";

type Workspace = "search" | "index" | "versions" | "update" | "history" | "evaluation";

type EvolutionResult = {
  rank: number;
  score: number;
  symbol?: string;
  path?: string;
  preview?: string;
  first_seen_commit?: string;
  last_seen_commit?: string;
  occurrences?: Array<{ commit?: string; path?: string; symbol?: string }>;
};

type EvolutionResponse = {
  repo: string;
  query: string;
  commits: string[];
  results: EvolutionResult[];
  runtime?: Record<string, unknown>;
};

const examples = [
  "How is the input preprocessed before the main function?",
  "Where is the Bluetooth settings deeplink used?",
  "Which functions call validate before save?",
  "Where is createSignature referenced?",
  "How is a payload signed?"
];

const frozenMetrics = {
  p0: [
    ["NDCG@10", "0.86950"],
    ["MRR@10", "0.84141"],
    ["HitRate@10", "95.56%"],
    ["Recall@100", "99.10%"]
  ],
  p1: [
    ["Avg embedding reuse", "86.83%"],
    ["Embedding work saved", "96.42%"],
    ["Max measured speedup", "15.54x"],
    ["Retrieval parity", "PASS"]
  ],
  evo: [
    ["Raw HitRate@5", "0.5000"],
    ["Grouped HitRate@5", "0.6667"],
    ["Raw MRR", "0.2778"],
    ["Grouped MRR", "0.3667"]
  ]
};

export function App() {
  const [workspace, setWorkspace] = useState<Workspace>("search");
  const [repoId, setRepoId] = useState("js_hands_on_smoke");
  const [commit, setCommit] = useState("");
  const [query, setQuery] = useState("Where is openBluetoothSettings used?");
  const [topK, setTopK] = useState(10);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [recent, setRecent] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [registerPath, setRegisterPath] = useState("");
  const [registerRef, setRegisterRef] = useState("HEAD");
  const [registerResult, setRegisterResult] = useState<Record<string, unknown> | null>(null);
  const [versionA, setVersionA] = useState("");
  const [versionB, setVersionB] = useState("");
  const [versionLeft, setVersionLeft] = useState<QueryResponse | null>(null);
  const [versionRight, setVersionRight] = useState<QueryResponse | null>(null);
  const [updateTarget, setUpdateTarget] = useState("");
  const [updateResult, setUpdateResult] = useState<Record<string, unknown> | null>(null);
  const [historyResult, setHistoryResult] = useState<EvolutionResponse | null>(null);

  useEffect(() => {
    refreshHealth();
  }, []);

  async function refreshHealth() {
    try {
      setHealth(await getJson<Record<string, unknown>>("/health"));
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : String(err));
    }
  }

  async function runQuery(targetCommit = commit) {
    setBusy(true);
    setPhase("Classifying query...");
    setError(null);
    const started = performance.now();
    try {
      setPhase("Searching semantic and structural indexes...");
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
      setError(friendlyError(err));
    } finally {
      setBusy(false);
      setPhase(null);
    }
  }

  async function registerRepo() {
    setBusy(true);
    setPhase("Indexing repository...");
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
      setError(friendlyError(err));
    } finally {
      setBusy(false);
      setPhase(null);
    }
  }

  async function compareVersions() {
    setBusy(true);
    setPhase("Searching both versions...");
    setError(null);
    try {
      const [left, right] = await Promise.all([
        postJson<QueryResponse>("/query", { repo_id: repoId, commit: versionA || undefined, query, top_k: topK }),
        postJson<QueryResponse>("/query", { repo_id: repoId, commit: versionB || undefined, query, top_k: topK })
      ]);
      setVersionLeft(left);
      setVersionRight(right);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setBusy(false);
      setPhase(null);
    }
  }

  async function updateRepo() {
    setBusy(true);
    setPhase("Updating index...");
    setError(null);
    try {
      const payload = await postJson<Record<string, unknown>>(`/repos/${repoId}/update`, { commit: updateTarget });
      setUpdateResult(payload);
      setCommit(String(payload.new_commit ?? ""));
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setBusy(false);
      setPhase(null);
    }
  }

  async function searchHistory() {
    setBusy(true);
    setPhase("Searching across history...");
    setError(null);
    try {
      setHistoryResult(await postJson<EvolutionResponse>("/search/evolution", {
        repo_id: repoId,
        query,
        top_k: topK,
        include_evolution_context: true
      }));
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setBusy(false);
      setPhase(null);
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.ctrlKey && event.key === "Enter") runQuery();
  }

  const nav = useMemo(() => [
    ["search", Search, "Search"],
    ["index", Boxes, "Index Repository"],
    ["versions", GitBranch, "Versions"],
    ["update", Zap, "Update Index"],
    ["history", History, "History"],
    ["evaluation", BarChart3, "Evaluation"]
  ] as const, []);

  const embeddingReady = Boolean((health?.llama_cpp as Record<string, unknown> | undefined)?.server_healthy);

  return (
    <div className="min-h-screen bg-graphite text-slate-100">
      <aside className="fixed inset-y-0 left-0 w-72 border-r border-line bg-[#0b1018] p-5">
        <div className="mb-7">
          <div className="text-xs uppercase tracking-[0.28em] text-samsung">Samsung Theme 1</div>
          <h1 className="mt-3 text-2xl font-semibold">Agentic Code Intelligence</h1>
          <p className="mt-2 text-sm leading-5 text-slate-400">Retrieve relevant code snippets from large codebases using natural-language queries.</p>
        </div>
        <nav className="space-y-2">
          {nav.map(([id, Icon, label]) => (
            <button key={id} onClick={() => setWorkspace(id)} className={`nav-item ${workspace === id ? "nav-active" : ""}`}>
              <Icon size={17} /> {label}
            </button>
          ))}
        </nav>
        <div className="absolute bottom-5 left-5 right-5 rounded-xl border border-line bg-panel p-3 text-xs text-slate-400">
          <div className="mb-2 flex items-center gap-2 text-slate-200"><Server size={14} /> System Status</div>
          <StatusLine label="Backend" ok={Boolean(health)} text={health ? "Ready" : "Offline"} />
          <StatusLine label="Embedding model" ok={embeddingReady} text={embeddingReady ? "Ready" : "Offline"} />
          <StatusLine label="Repository index" ok={Boolean(health)} text={health ? "Ready" : "Unknown"} />
          {healthError && <div className="mt-2 text-red-300">Backend offline. Start FastAPI and retry.</div>}
          <button className="mt-3 text-samsung hover:text-blue-300" onClick={refreshHealth}>Refresh status</button>
        </div>
      </aside>

      <main className="ml-72 p-5">
        {error && <ErrorBanner message={error} />}
        {busy && phase && <div className="mb-4 flex items-center gap-2 rounded-xl border border-blue-400/20 bg-blue-400/10 p-3 text-sm text-blue-100"><Loader2 size={16} className="animate-spin" /> {phase}</div>}

        {workspace === "search" && (
          <section className="space-y-4">
            <SearchHeader loadExample={() => { setRepoId("js_hands_on_smoke"); setQuery("Which functions call validate before save?"); setTopK(10); }} />
            <SearchForm repoId={repoId} setRepoId={setRepoId} commit={commit} setCommit={setCommit} query={query} setQuery={setQuery} topK={topK} setTopK={setTopK} onSearch={() => runQuery()} onKeyDown={onKeyDown} busy={busy} />
            <ExampleQueries setQuery={setQuery} />
            <TraceAndResults response={response} />
            <HistoryBox recent={recent} setQuery={setQuery} />
          </section>
        )}

        {workspace === "index" && (
          <Panel title="Index Repository" subtitle="Register any local Git repository. The backend builds manifests, chunks, embeddings, a vector index, and a structural graph." icon={<Boxes size={19} />}>
            <div className="grid gap-3 md:grid-cols-[1fr_2fr_1fr]">
              <Input label="Repository ID" value={repoId} onChange={setRepoId} />
              <Input label="Local repository path" value={registerPath} onChange={setRegisterPath} placeholder="C:\\path\\to\\repo" />
              <Input label="Git ref / commit" value={registerRef} onChange={setRegisterRef} />
            </div>
            <button className="primary mt-4" disabled={busy || !registerPath} onClick={registerRepo}><Boxes size={16} /> Index Repository</button>
            <IndexingResult value={registerResult} />
          </Panel>
        )}

        {workspace === "versions" && (
          <Panel title="Retrieval Across Versions" subtitle="Run the same arbitrary query against two indexed commits." icon={<GitBranch size={19} />}>
            <SearchForm repoId={repoId} setRepoId={setRepoId} commit={commit} setCommit={setCommit} query={query} setQuery={setQuery} topK={topK} setTopK={setTopK} onSearch={compareVersions} onKeyDown={onKeyDown} busy={busy} />
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <Input label="Commit A" value={versionA} onChange={setVersionA} placeholder="first commit/ref" />
              <Input label="Commit B" value={versionB} onChange={setVersionB} placeholder="second commit/ref" />
            </div>
            <button className="primary mt-4" disabled={busy} onClick={compareVersions}><GitBranch size={16} /> Compare Versions</button>
            <div className="mt-5 grid gap-4 xl:grid-cols-2">
              <VersionColumn label="Commit A" commit={versionLeft?.commit ?? versionA} response={versionLeft} />
              <VersionColumn label="Commit B" commit={versionRight?.commit ?? versionB} response={versionRight} />
            </div>
          </Panel>
        )}

        {workspace === "update" && (
          <Panel title="Update Version" subtitle="Move an indexed repository from one commit to another and show incremental reuse evidence." icon={<Zap size={19} />}>
            <div className="grid gap-3 md:grid-cols-3">
              <Input label="Repository ID" value={repoId} onChange={setRepoId} />
              <Input label="Current commit" value={commit} onChange={setCommit} />
              <Input label="Target commit" value={updateTarget} onChange={setUpdateTarget} />
            </div>
            <button className="primary mt-4" disabled={busy || !updateTarget} onClick={updateRepo}><Zap size={16} /> Update Index</button>
            <UpdateResult value={updateResult} />
          </Panel>
        )}

        {workspace === "history" && (
          <Panel title="Search Across History" subtitle="Return distinct semantic code states across indexed versions." icon={<History size={19} />}>
            <SearchForm repoId={repoId} setRepoId={setRepoId} commit={commit} setCommit={setCommit} query={query} setQuery={setQuery} topK={topK} setTopK={setTopK} onSearch={searchHistory} onKeyDown={onKeyDown} busy={busy} />
            <button className="primary mt-4" disabled={busy || !query.trim()} onClick={searchHistory}><History size={16} /> Search Across History</button>
            <EvolutionResults response={historyResult} />
          </Panel>
        )}

        {workspace === "evaluation" && (
          <Panel title="Evaluation Evidence" subtitle="Supporting metrics are separated from the retrieval workflow." icon={<BarChart3 size={19} />}>
            <MetricSection title="Official P0 - AppsRetrieval" subtitle="Official MTEB AppsRetrieval test split" rows={frozenMetrics.p0} />
            <MetricSection title="P1 evidence" subtitle="Real Git-history benchmark repository; benchmark evidence, not live runtime." rows={frozenMetrics.p1} />
            <MetricSection title="Evolutionary Retrieval" subtitle="Internal validation - not Samsung official metric" rows={frozenMetrics.evo} />
          </Panel>
        )}
      </main>
    </div>
  );
}

function SearchHeader({ loadExample }: { loadExample: () => void }) {
  return (
    <div className="rounded-2xl border border-line bg-[#0d131d] p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold">Agentic Code Intelligence</h2>
          <p className="mt-1 text-sm text-slate-400">Retrieve relevant code snippets from large codebases using natural-language queries.</p>
        </div>
        <button className="secondary" onClick={loadExample}>Load Example</button>
      </div>
    </div>
  );
}

function SearchForm(props: { repoId: string; setRepoId: (v: string) => void; commit: string; setCommit: (v: string) => void; query: string; setQuery: (v: string) => void; topK: number; setTopK: (v: number) => void; onSearch: () => void; onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void; busy: boolean }) {
  return (
    <div className="rounded-2xl border border-line bg-panel p-4">
      <div className="grid gap-3 md:grid-cols-[1fr_1fr_120px]">
        <Input label="Repository" value={props.repoId} onChange={props.setRepoId} placeholder="sample_repo" />
        <Input label="Version / Commit" value={props.commit} onChange={props.setCommit} placeholder="active indexed commit" />
        <label className="block text-sm text-slate-300">Top K
          <select className="input mt-1" value={props.topK} onChange={(e) => props.setTopK(Number(e.target.value))}>
            <option>5</option>
            <option>10</option>
            <option>20</option>
          </select>
        </label>
      </div>
      <label className="mt-4 block text-sm text-slate-300">Natural-language query</label>
      <textarea className="input mt-1 h-24 resize-none" value={props.query} onChange={(e) => props.setQuery(e.target.value)} onKeyDown={props.onKeyDown} placeholder="How is the input preprocessed before the main function?" />
      <div className="mt-4 flex items-center gap-3">
        <button className="primary" disabled={props.busy || !props.query.trim()} onClick={props.onSearch}>{props.busy ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />} Search Code</button>
        <span className="text-xs text-slate-500">Ctrl+Enter also runs search.</span>
      </div>
    </div>
  );
}

function TraceAndResults({ response }: { response: QueryResponse | null }) {
  if (!response) return <EmptyState />;
  const total = Number(response.timing.total_ms ?? response.timing.browser_roundtrip_ms ?? 0);
  return (
    <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
      <aside className="space-y-4">
        <div className="rounded-2xl border border-line bg-panel p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 font-medium"><Activity size={16} /> Agentic Retrieval Trace</div>
            <span className="badge">{labelCase(response.query_type)}</span>
          </div>
          <TraceList trace={response.trace} />
        </div>
        <div className="rounded-2xl border border-line bg-panel p-4">
          <div className="flex items-center gap-2 text-sm font-medium"><Clock3 size={16} /> Speed</div>
          <div className="mt-3 text-2xl font-semibold">{Math.round(total)} ms</div>
          <div className="text-sm text-slate-400">Retrieved {response.results.length} snippets</div>
          <details className="mt-3">
            <summary className="cursor-pointer text-sm text-samsung">Stage timing breakdown</summary>
            <TimingGrid timing={response.timing} />
          </details>
        </div>
      </aside>
      <ResultColumn title="Ranked code snippets" response={response} />
    </div>
  );
}

function ResultCard({ result }: { result: QueryResult }) {
  const calls = result.ordered_calls?.map((call) => call.name ?? call.raw).filter(Boolean);
  return (
    <article className="rounded-xl border border-line bg-[#0a0f17] p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="rank">#{result.rank}</span>
        <span className="font-medium text-slate-100">{result.symbol ?? "anonymous"}</span>
        <span className="badge">{labelCase(result.evidence_type ?? "semantic")}</span>
        <span className="ml-auto text-slate-400">score {formatScore(result.score)}</span>
      </div>
      <div className="mt-2 font-mono text-xs text-samsung">{result.path}:{result.start_line ?? "?"}-{result.end_line ?? "?"}</div>
      <div className="mt-1 text-xs text-slate-500">{result.symbol_type ?? "code"} · commit {shortSha(result.commit)}</div>
      {result.referenced_symbol && <div className="mt-2 text-xs text-slate-300">Evidence: references <span className="font-mono text-blue-200">{result.referenced_symbol}</span></div>}
      {calls?.length ? <div className="mt-3 rounded-lg border border-blue-400/20 bg-blue-400/5 p-2 text-xs text-blue-100"><div className="mb-1 font-medium">Static structural/source-order evidence</div><div className="font-mono">{calls.join(" -> ")}</div><div className="mt-1 text-slate-400">{result.limitation ?? "Static source-order evidence; not runtime execution certainty."}</div></div> : null}
      <details open={result.rank <= 3} className="mt-3">
        <summary className="cursor-pointer text-xs text-slate-400">Code snippet</summary>
        <pre className="code mt-2">{withLineNumbers(result.snippet ?? "", result.start_line ?? 1)}</pre>
      </details>
    </article>
  );
}

function VersionColumn({ label, commit, response }: { label: string; commit?: string; response: QueryResponse | null }) {
  return <div className="rounded-xl border border-line bg-panel p-4"><div className="mb-3 text-sm text-slate-400">Searching version: <span className="font-mono text-slate-200">{shortSha(commit)}</span></div><ResultColumn title={label} response={response} /></div>;
}

function UpdateResult({ value }: { value: Record<string, unknown> | null }) {
  if (!value) return <InfoBox text="Run Update Index to see changed files, reused chunks, reused embeddings, generated embeddings, vectors, and update time." />;
  const changed = value.changed_files as Record<string, unknown> | undefined;
  return <div className="mt-4 rounded-2xl border border-line bg-panel p-4"><h3 className="mb-3 font-medium">Incremental update result</h3><div className="grid gap-3 md:grid-cols-4"><Stat label="Unchanged files" value={changed?.unchanged} /><Stat label="Modified files" value={changed?.modified} /><Stat label="Added files" value={changed?.added} /><Stat label="Deleted files" value={changed?.deleted} /><Stat label="Reused chunks" value={value.reusable_chunks} /><Stat label="Reused embeddings" value={value.embeddings_reused} /><Stat label="Generated embeddings" value={value.embeddings_generated} /><Stat label="Update time" value={`${formatMs(secondsToMs(value.update_runtime_seconds))} ms`} /></div></div>;
}

function IndexingResult({ value }: { value: Record<string, unknown> | null }) {
  if (!value) return <InfoBox text="Index a local JavaScript repository to enable immediate querying. No hardcoded repository is required." />;
  return <div className="mt-4 rounded-2xl border border-line bg-panel p-4"><h3 className="mb-3 font-medium">Indexing output</h3><div className="grid gap-3 md:grid-cols-4"><Stat label="Source files" value={value.source_files} /><Stat label="LOC" value={value.loc} /><Stat label="Symbols" value={value.symbols} /><Stat label="Chunks" value={value.chunks} /><Stat label="Call edges" value={value.call_edges} /><Stat label="Reference edges" value={value.reference_edges} /><Stat label="Embeddings reused" value={value.reused_embeddings} /><Stat label="Embeddings generated" value={value.newly_generated_embeddings} /><Stat label="Indexing time" value={`${formatMs(secondsToMs(value.indexing_runtime_seconds))} ms`} /></div></div>;
}

function EvolutionResults({ response }: { response: EvolutionResponse | null }) {
  if (!response) return <InfoBox text="Search arbitrary questions across indexed history. Unchanged duplicate code states are grouped." />;
  return <div className="mt-4 rounded-2xl border border-line bg-panel p-4"><div className="mb-3 flex items-center justify-between"><h3 className="font-medium">Distinct semantic states</h3><span className="text-xs text-slate-500">{response.results.length} grouped results</span></div><div className="space-y-3">{response.results.map((item) => <article key={`${item.rank}-${item.path}-${item.symbol}`} className="rounded-xl border border-line bg-[#0a0f17] p-3"><div className="flex items-center gap-2 text-sm"><span className="rank">#{item.rank}</span><span className="font-medium">{item.symbol ?? "state"}</span><span className="ml-auto text-slate-400">score {formatScore(item.score)}</span></div><div className="mt-2 font-mono text-xs text-samsung">{item.path}</div><div className="mt-1 text-xs text-slate-500">first {shortSha(item.first_seen_commit)} · last {shortSha(item.last_seen_commit)}</div><div className="mt-2 text-xs text-slate-400">Occurrence commits: {(item.occurrences ?? []).map((o) => shortSha(o.commit)).join(", ") || "n/a"}</div><pre className="code mt-3">{item.preview ?? "No preview returned."}</pre></article>)}</div></div>;
}

function TraceList({ trace }: { trace: Array<Record<string, unknown>> }) {
  return <div className="space-y-2 text-sm">{trace.map((step, idx) => <div key={idx} className="rounded-lg bg-black/20 p-2"><div className="flex items-center gap-2"><CheckCircle2 size={14} className="text-blue-300" /> {traceText(step)}</div></div>)}</div>;
}

function TimingGrid({ timing }: { timing: Record<string, number | string | null> }) {
  const entries = Object.entries(timing).filter(([, value]) => value !== null && value !== undefined);
  return <div className="mt-3 grid gap-2 text-xs">{entries.map(([key, value]) => <div key={key} className="flex justify-between gap-3 rounded bg-black/20 px-2 py-1"><span className="text-slate-400">{key}</span><span className="font-mono text-slate-200">{typeof value === "number" ? `${formatMs(value)} ms` : String(value)}</span></div>)}</div>;
}

function ResultColumn({ title, response }: { title: string; response: QueryResponse | null }) {
  return <div className="rounded-2xl border border-line bg-panel p-4"><h3 className="mb-3 font-medium">{title}</h3>{response?.results?.length ? <div className="space-y-3">{response.results.map((result) => <ResultCard key={`${result.rank}-${result.path}-${result.start_line}-${result.symbol}`} result={result} />)}</div> : <div className="text-sm text-slate-500">No results yet.</div>}</div>;
}

function Panel({ title, subtitle, icon, children }: { title: string; subtitle?: string; icon: ReactNode; children: ReactNode }) {
  return <section className="rounded-2xl border border-line bg-[#0d131d] p-5"><h2 className="flex items-center gap-2 text-xl font-semibold">{icon}{title}</h2>{subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}<div className="mt-4">{children}</div></section>;
}

function Input({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return <label className="block text-sm text-slate-300">{label}<input className="input mt-1" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} /></label>;
}

function ExampleQueries({ setQuery }: { setQuery: (v: string) => void }) {
  return <div className="flex flex-wrap gap-2">{examples.map((item) => <button key={item} className="chip" onClick={() => setQuery(item)}>{item}</button>)}</div>;
}

function MetricSection({ title, subtitle, rows }: { title: string; subtitle: string; rows: string[][] }) {
  return <div className="mb-5 rounded-xl border border-line bg-panel p-4"><h3 className="font-medium">{title}</h3><p className="text-sm text-slate-400">{subtitle}</p><div className="mt-3 grid gap-3 md:grid-cols-4">{rows.map(([k, v]) => <Stat key={k} label={k} value={v} />)}</div></div>;
}

function Stat({ label, value }: { label: string; value: unknown }) {
  return <div className="rounded-lg bg-black/20 p-3"><div className="text-xs text-slate-500">{label}</div><div className="mt-1 text-lg font-semibold text-slate-100">{displayValue(value)}</div></div>;
}

function StatusLine({ label, ok, text }: { label: string; ok: boolean; text: string }) {
  return <div className="flex justify-between gap-2"><span>{label}</span><span className={ok ? "text-emerald-300" : "text-red-300"}>{text}</span></div>;
}

function ErrorBanner({ message }: { message: string }) {
  return <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-100">{message}</div>;
}

function EmptyState() {
  return <div className="rounded-2xl border border-line bg-panel p-6 text-sm text-slate-400">Enter any natural-language query and run Search Code. Results will show ranked snippets with exact file and line locations.</div>;
}

function InfoBox({ text }: { text: string }) {
  return <div className="mt-4 rounded-xl border border-line bg-black/20 p-4 text-sm text-slate-400">{text}</div>;
}

function HistoryBox({ recent, setQuery }: { recent: string[]; setQuery: (v: string) => void }) {
  if (!recent.length) return null;
  return <div className="rounded-2xl border border-line bg-panel p-4"><h3 className="mb-2 text-sm font-medium">Session query history</h3><div className="flex flex-wrap gap-2">{recent.map((item) => <button className="chip" key={item} onClick={() => setQuery(item)}>{item}</button>)}</div></div>;
}

function traceText(step: Record<string, unknown>) {
  const stage = String(step.stage ?? "stage");
  if (step.skipped) return `${stage}: skipped (${String(step.reason ?? "not needed")})`;
  if (step.result) return `Query classified: ${labelCase(String(step.result))}`;
  if (step.candidates) return `${stage}: ${String(step.candidates)} candidates`;
  if (step.matches) return `${stage}: ${String(step.matches)} matches`;
  if (step.regions) return `Read candidate regions: ${String(step.regions)}`;
  if (step.remaining_candidates) return `Refined candidates: ${String(step.remaining_candidates)}`;
  if (step.returned) return `Final ranked results: ${String(step.returned)}`;
  return stage;
}

function withLineNumbers(snippet: string, start: number) {
  return snippet.split("\n").map((line, idx) => `${String(start + idx).padStart(4, " ")} | ${line}`).join("\n");
}

function labelCase(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function shortSha(value?: string) {
  return value ? value.slice(0, 8) : "active";
}

function formatScore(value?: number) {
  return Number(value ?? 0).toFixed(4);
}

function formatMs(value: number) {
  return Number(value).toFixed(value >= 10 ? 0 : 2);
}

function secondsToMs(value: unknown) {
  return typeof value === "number" ? value * 1000 : 0;
}

function friendlyError(err: unknown) {
  const message = err instanceof Error ? err.message : String(err);
  if (message.toLowerCase().includes("embedding")) return "Embedding server unavailable. Start llama.cpp and retry.";
  if (message.toLowerCase().includes("not registered")) return "Repository is not indexed yet. Use Index Repository first.";
  if (message.toLowerCase().includes("commit")) return "Invalid or unindexed commit. Check the repository version and retry.";
  return message;
}

function displayValue(value: unknown) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}
