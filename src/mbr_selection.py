import math
import numpy as np
import sacrebleu
from typing import List

class MBRSelector:
    def __init__(
        self,
        pool_cap: int = 32,
        w_chrf: float = 0.55,
        w_bleu: float = 0.25,
        w_jaccard: float = 0.20,
        w_length: float = 0.10,
    ):
        self._chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
        self.pool_cap = pool_cap
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length = w_length
        self._pw_total = max(w_chrf + w_bleu + w_jaccard, 1e-9)

    def _chrfpp(self, a: str, b: str) -> float:
        if not a or not b: return 0.0
        return float(self._chrf_metric.sentence_score(a, [b]).score)

    def _bleu(self, a: str, b: str) -> float:
        if not a or not b: return 0.0
        try: return float(self._bleu_metric.sentence_score(a, [b]).score)
        except Exception: return 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb: return 100.0
        if not ta or not tb: return 0.0
        return 100.0 * len(ta & tb) / len(ta | tb)

    def _pairwise_score(self, a: str, b: str) -> float:
        s = (self.w_chrf * self._chrfpp(a, b) + self.w_bleu * self._bleu(a, b) + self.w_jaccard * self._jaccard(a, b))
        return s / self._pw_total

    @staticmethod
    def _length_bonus(lengths: List[int], idx: int) -> float:
        if len(lengths) == 0: return 100.0
        median = float(np.median(lengths))
        sigma = max(median * 0.4, 5.0)
        z = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    @staticmethod
    def _dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def pick(self, candidates: List[str]) -> str:
        cands = self._dedup(candidates)
        if self.pool_cap: cands = cands[:self.pool_cap]
        n = len(cands)
        
        if n == 0: return ""
        if n == 1: return cands[0]

        lengths = [len(c.split()) for c in cands]
        scores = []

        for i in range(n):
            pw = sum(self._pairwise_score(cands[i], cands[j]) for j in range(n) if j != i) / max(1, n - 1)
            lb = self._length_bonus(lengths, i)
            total = pw + self.w_length * lb
            scores.append(total)

        return cands[int(np.argmax(scores))]