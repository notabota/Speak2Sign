import argparse, json, os, sys
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

def load_glosser(glosser_dir: str):
    sys.path.append(os.path.abspath(glosser_dir))
    import glosser as G
    return G.ASLGlosser(os.path.join(glosser_dir, "data"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--glosser-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_text")
    g.add_argument("--input_file")
    ap.add_argument("--pred_file")
    args = ap.parse_args()

    tok = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    def generate_text(inp: str) -> str:
        ids = tok(inp, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=64, num_beams=4)
        return tok.decode(out[0], skip_special_tokens=True)

    if args.input_text:
        glosser = load_glosser(args.glosser_dir)
        rules = glosser.gloss(args.input_text).gloss
        inp = f"ENGLISH: {args.input_text}\nRULES: {rules}"
        print("RULES:", rules)
        print("MODEL:", generate_text(inp))
    else:
        ds = load_dataset("json", data_files=args.input_file, split="train")
        hyps = [generate_text(ex["input"]) for ex in ds]
        if args.pred_file:
            with open(args.pred_file, "w", encoding="utf-8") as f:
                for h in hyps: f.write(h + "\n")
        else:
            for h in hyps[:10]: print(h)

if __name__ == "__main__":
    main()
