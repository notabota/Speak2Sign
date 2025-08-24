import argparse
import sacrebleu

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="references, one per line")
    ap.add_argument("--hyps", required=True, help="hypotheses, one per line")
    args = ap.parse_args()

    refs = read_lines(args.refs)
    hyps = read_lines(args.hyps)
    assert len(refs) == len(hyps), "Mismatch in lines"

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    print(f"BLEU: {bleu.score:.2f}")
    print(f"chrF: {chrf.score:.2f}")

if __name__ == "__main__":
    main()
