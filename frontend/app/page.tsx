"use client";
import { useState } from "react";

/* ---------- Types ---------- */
type ModelKey = "valhalla/distilbart-mnli-12-1" | "xlm-roberta-xnli";

interface LabelScore {
  label: string;
  score: number;
}

interface ClassificationItem {
  text: string;
  picked: LabelScore[];
  all: LabelScore[];
}

interface ClassifyResponse {
  results: ClassificationItem[];
}

/* ---------- Config ---------- */
const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ---------- Component ---------- */
export default function Home() {
  const [texts, setTexts] = useState([].join("\n"));

  const [labels, setLabels] = useState(
    "toxic, insult, harassment, hate_speech, racism, sexism, sexual_content, self_harm, spam, safe"
  );
  const [multi, setMulti] = useState<boolean>(true);
  const [thresh, setThresh] = useState<number>(0.7);
  const [model, setModel] = useState<ModelKey>("valhalla/distilbart-mnli-12-1");
  const [results, setResults] = useState<ClassificationItem[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  async function classify(): Promise<void> {
    setLoading(true);
    setError(null);
    setResults(null);

    const payload = {
      texts: texts
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      labels: labels
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      multi_label: multi,
      threshold: thresh,
      model,
    };

    try {
      const res = await fetch(`${apiBase}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data: ClassifyResponse = await res.json();
      setResults(data.results);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Request failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  function downloadCSV(): void {
    if (!results) return;
    const rows: string[][] = [["text", "predicted_labels", "scores"]];
    results.forEach((r) => {
      const labs = r.picked.map((p) => p.label).join(" | ");
      const scs = r.picked.map((p) => p.score.toFixed(3)).join(" | ");
      rows.push([r.text, labs, scs]);
    });
    const csv = rows
      .map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "moderation_results.csv";
    a.click();
  }

  return (
    <main className="mx-auto max-w-3xl p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Content Moderation Tool</h1>
        <p className="text-gray-500">
          Detect offensive, hateful, or unsafe messages.
        </p>
      </header>

      <section className="space-y-2">
        <label className="text-sm font-medium">Messages (one per line)</label>
        <textarea
          value={texts}
          placeholder="Enter messages here, one per line..."
          onChange={(e) => setTexts(e.target.value)}
          className="w-full h-40 rounded-md border p-3 focus:outline-none focus:ring"
        />
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <label className="text-sm font-medium">
            Custom labels (comma-separated)
          </label>
          <textarea
            value={labels}
            onChange={(e) => setLabels(e.target.value)}
            className="w-full min-h-[60px] rounded-md border p-3 focus:outline-none focus:ring resize-y"
            placeholder="toxic, insult, harassment, hate_speech, ..."
          />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as ModelKey)}
            className="w-full rounded-md border p-3"
          >
            <option value="bart-large-mnli">BART-MNLI (English)</option>
            <option value="xlm-roberta-xnli">XLM-R XNLI (Multilingual)</option>
          </select>
        </div>
      </section>

      <section className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={multi}
            onChange={(e) => setMulti(e.target.checked)}
          />
          <span>Multi-label</span>
        </label>
        <div className="flex items-center gap-3">
          <span className="text-sm">Threshold: {thresh.toFixed(2)}</span>
          <input
            type="range"
            min={0.1}
            max={0.95}
            step={0.01}
            value={thresh}
            onChange={(e) => setThresh(parseFloat(e.target.value))}
          />
        </div>
        <button
          onClick={classify}
          className="rounded-md bg-indigo-600 px-4 py-2 text-white disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Classifying..." : "Classify"}
        </button>
        {results && (
          <button onClick={downloadCSV} className="rounded-md border px-4 py-2">
            Download CSV
          </button>
        )}
      </section>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 text-red-800">
          {error}
        </div>
      )}

      {results && (
        <section className="space-y-3">
          <h2 className="font-semibold">Results</h2>
          <div className="overflow-hidden rounded-md border">
            <table className="w-full table-fixed">
              <thead className="bg-gray-50">
                <tr>
                  <th className="w-[60%] p-2 text-left text-sm font-medium">
                    Message
                  </th>
                  <th className="w-[25%] p-2 text-left text-sm font-medium">
                    Predicted Category/Categories
                  </th>
                  <th className="w-[15%] p-2 text-left text-sm font-medium">
                    Score(s)
                  </th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-t align-top">
                    <td className="p-2 whitespace-pre-wrap break-words">
                      {r.text}
                    </td>
                    <td className="p-2 whitespace-pre-wrap break-words">
                      {r.picked.map((p) => p.label).join(", ")}
                    </td>
                    <td className="p-2 whitespace-pre-wrap break-words">
                      {r.picked.map((p) => p.score.toFixed(3)).join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <details className="rounded-md border p-3">
            <summary className="cursor-pointer">Show raw scores (JSON)</summary>
            <pre className="mt-2 whitespace-pre-wrap text-sm">
              {JSON.stringify(results, null, 2)}
            </pre>
          </details>
        </section>
      )}
    </main>
  );
}
