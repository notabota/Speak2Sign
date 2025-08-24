import argparse, csv, json, os, sys, random
import rules.glosser as G

def load_glosser(glosser_dir: str):
    sys.path.append(os.path.abspath(glosser_dir))
    return G.ASLGlosser(os.path.join(glosser_dir, "data"))

def split_rows(rows, train=0.8, dev=0.1, seed=42):
    random.Random(seed).shuffle(rows)
    n = len(rows); n_tr = int(n*train); n_dev = int(n*dev)
    return rows[:n_tr], rows[n_tr:n_tr+n_dev], rows[n_tr+n_dev:]

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: english,gloss")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--glosser-dir", default="glosser")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    glosser = load_glosser(args.glosser_dir)

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            eng = r["english"].strip()
            tgt = r["gloss"].strip()
            rules = glosser.gloss(eng).gloss
            rows.append({
                "input": f"ENGLISH: {eng}\nRULES: {rules}",
                "target": tgt
            })

    train, dev, test = split_rows(rows, seed=args.seed)

    write_jsonl(args.train, train)
    write_jsonl(args.dev, dev)
    write_jsonl(args.test, test)

if __name__ == "__main__":
    main()
