import json, os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class GlossResult:
    gloss_tokens: List[str]
    gloss: str
    sentence_nmm: Dict[str, Optional[str]] = field(default_factory=dict)

class ASLGlosser:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "lexicon.json"), "r", encoding="utf-8") as f:
            self.LEXICON: Dict[str, Optional[str]] = json.load(f)
        with open(os.path.join(data_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(data_dir, "phrases.json"), "r", encoding="utf-8") as f:
            phr = json.load(f)
        with open(os.path.join(data_dir, "contractions.json"), "r", encoding="utf-8") as f:
            self.CONTRACTIONS = json.load(f)

        self.FUNCTION_WORDS = set(cfg["function_words"])
        self.WH = set(cfg["wh_words"])
        self.AUX_STARTERS = set(cfg["aux_starters"])
        self.VERB_LIKE = set(cfg["verb_like"])
        self.UNIT_MAP = cfg["units_map"]
        self.FRONT_TIME = set(cfg["front_time"])

        self.MWE_REPLACE: Dict[str,str] = phr["mwe_replace"]
        self.PRE_INTENS = set(phr["pre_intensifiers"])
        self.POST_INTENS = set(phr["post_intensifiers"])
        self.PERFECT_TRIGGERS = phr["perfect_triggers"]
        self.AM_PM = set(phr["am_pm"])

        # Regex for tokenization
        self._tok_re = re.compile(r"[A-Za-z]+|\d+|[^\w\s]", flags=re.UNICODE)

    # ---------- helpers ----------
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions using patterns from contractions.json.
        Matches both straight (') and curly (’) apostrophes, case-insensitive, word-bounded.
        """
        out = text
        for pair in self.CONTRACTIONS:
            if isinstance(pair, list) and len(pair) == 2:
                pat, rep = pair
            elif isinstance(pair, dict):
                pat, rep = pair.get("pattern"), pair.get("replace")
            else:
                continue
            if not pat or rep is None:
                continue
            esc = re.escape(pat)
            # allow both types of apostrophes
            esc = esc.replace("\\'", "['’']").replace("’", "['’]")
            out = re.sub(rf"\b{esc}\b", rep, out, flags=re.IGNORECASE)
        return out

    def tokenize(self, text: str):
        return self._tok_re.findall(text)

    def detect_qtype(self, alpha_lower, original_text: str) -> Optional[str]:
        if any(w in self.WH for w in alpha_lower):
            return "WH"
        if original_text.strip().endswith("?") or (alpha_lower and alpha_lower[0] in self.AUX_STARTERS):
            return "YN"
        return None

    def detect_negation(self, alpha_lower) -> bool:
        return any(w in {"not","never","no"} for w in alpha_lower)

    # ---------- normalizers ----------
    def normalize_mwe(self, toks: List[str]) -> List[str]:
        text = " ".join(toks)
        for k, v in self.MWE_REPLACE.items():
            text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
        return text.split()

    def morph_normalize(self, toks: List[str]) -> List[str]:
        out = []
        for w in toks:
            base = w
            # past
            if w.endswith("ied"):
                cand = w[:-3] + "y"
                if cand in self.LEXICON: base = cand
            elif w.endswith("ed"):
                cand = w[:-2]
                if cand in self.LEXICON: base = cand
                else:
                    cand2 = w[:-1]
                    if cand2 in self.LEXICON: base = cand2
            # progressive
            if w.endswith("ing"):
                cand = w[:-3]; cand_e = cand + "e"
                if cand in self.LEXICON: base = cand
                elif cand_e in self.LEXICON: base = cand_e
            # plural
            if w.endswith("es"):
                cand = w[:-2]
                if cand in self.LEXICON: base = cand
            elif w.endswith("s") and len(w) > 3:
                cand = w[:-1]
                if cand in self.LEXICON: base = cand
            out.append(base)
        return out

    def normalize_perfect(self, toks: List[str]) -> List[str]:
        out=[]; i=0
        triggers = set()
        for vals in self.PERFECT_TRIGGERS.values():
            for v in vals: triggers.add(v)
        while i < len(toks):
            w = toks[i]; nxt = toks[i+1] if i+1 < len(toks) else ""
            if w in {"have","has","had"} and nxt and (nxt in triggers or nxt.endswith("ed")):
                out.append("FINISH"); i += 1; continue
            out.append(w); i += 1
        return out

    def number_units(self, toks: List[str]) -> List[str]:
        out=[]; i=0
        while i < len(toks):
            w=toks[i]; wnext=toks[i+1] if i+1 < len(toks) else ""
            if w.isdigit() and wnext in self.UNIT_MAP:
                out.append(f"{w}-{self.UNIT_MAP[wnext]}"); i += 2; continue
            out.append(w); i += 1
        return out

    # ---------- mapping ----------
    def map_token(self, tok: str, original_tok: str) -> Optional[str]:
        if tok in self.LEXICON:
            val = self.LEXICON[tok]
            if val is None: return None
            return val
        if tok in self.FUNCTION_WORDS: return None
        if re.fullmatch(r"\d+", tok): return tok
        if "-" in tok and tok.upper() == tok: return tok
        if re.search(r"[A-Z]", original_tok):
            up = re.sub(r"[^A-Za-z]", "", original_tok).upper()
            return f"FS-{up}" if up else None
        if tok.isalpha(): return f"FS-{tok.upper()}"
        return None

    # ---------- ordering/cleanup ----------
    def front_time_topic(self, gloss_tokens: List[str]) -> List[str]:
        front = [g for g in gloss_tokens if g in self.FRONT_TIME]
        rest = [g for g in gloss_tokens if g not in self.FRONT_TIME]
        return front + rest

    def move_wh_to_end(self, tokens: List[str]) -> List[str]:
        wh = [t for t in tokens if t in {"WHO","WHAT","WHERE","WHEN","WHY","HOW"}]
        rest = [t for t in tokens if t not in {"WHO","WHAT","WHERE","WHEN","WHY","HOW"}]
        return rest + wh

    def place_negation(self, tokens: List[str]) -> List[str]:
        pron = {"ME","YOU","WE","HE","SHE","THEY","IT"}
        for i,t in enumerate(tokens):
            if t in pron: return tokens[:i+1] + ["NOT"] + tokens[i+1:]
        return ["NOT"] + tokens

    def collapse_duplicates(self, toks: List[str]) -> List[str]:
        out=[]
        for t in toks:
            if out and out[-1] == t: continue
            out.append(t)
        return out

    # ---------- main ----------
    def gloss(self, text: str) -> GlossResult:
        expanded = self.expand_contractions(text)
        orig_toks = self.tokenize(expanded)
        alpha_lower = [t.lower() for t in orig_toks if t.isalpha()]

        qtype = self.detect_qtype(alpha_lower, expanded)
        neg = self.detect_negation(alpha_lower)

        words = [t.lower() for t in orig_toks if t.isalpha() or t.isdigit()]
        words = self.normalize_mwe(words)
        words = self.normalize_perfect(words)
        words = self.morph_normalize(words)
        words = self.number_units(words)

        mapped=[]; plus_next=False
        orig_alpha=[t for t in orig_toks if (t.isalpha() or t.isdigit())]
        orig_idx=0; i=0
        while i < len(words):
            w = words[i]
            orig_tok = orig_alpha[orig_idx] if orig_idx < len(orig_alpha) else w

            # post-intensifier "a lot"
            if i+1 < len(words) and (" ".join(words[i:i+2]) in self.POST_INTENS):
                if mapped: mapped[-1] = mapped[-1] + "++"
                i += 2; orig_idx += 2; continue

            # pre-intensifier
            if w in self.PRE_INTENS:
                plus_next=True; i += 1; orig_idx += 1; continue

            if w.isalpha() and w.upper()==w:
                g = w
            elif w in self.AM_PM:
                if mapped and re.fullmatch(r"\d+", mapped[-1]): g = w.upper()
                else: g=None
            else:
                g = self.map_token(w, orig_tok)

            if g:
                if plus_next: g = g + "++"; plus_next=False
                mapped.append(g)

            i += 1; orig_idx += 1

        mapped = self.front_time_topic(mapped)
        if neg and "NOT" not in mapped:
            mapped = self.place_negation(mapped)

        sent_nmm = {"brows": None, "head": "shake" if neg else None, "qtype": None}
        if qtype == "WH":
            mapped = self.move_wh_to_end(mapped); mapped.append("?")
            sent_nmm["brows"]="furrow"; sent_nmm["qtype"]="WH"
        elif qtype == "YN":
            mapped.append("?")
            sent_nmm["brows"]="raise"; sent_nmm["qtype"]="YN"

        mapped = self.collapse_duplicates(mapped)
        gloss_str = " ".join([t for t in mapped if t]).strip()
        return GlossResult(gloss_tokens=mapped, gloss=gloss_str, sentence_nmm=sent_nmm)

def main():
    data_dir = "data"
    text = 'I don’t like pizza at all.'
    res = ASLGlosser(data_dir).gloss(text)
    print(res.gloss)
    if any(res.sentence_nmm.values()):
        print("NMM:", res.sentence_nmm)

if __name__ == "__main__":
    main()
