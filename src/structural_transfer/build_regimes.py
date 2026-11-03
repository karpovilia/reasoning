"""HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013: regime files and rosters of the nine training variants.

A regime file holds the split code of every position and chunk; inputs and targets are the shared file of the corrected
ablation repeat and are not copied.  Codes: 0 train, 1 validation, 2 original test part (unchanged), 3 held-out group in
the original train part, 5 held-out group in the original validation part, 4 related group (R1 distillations) taken out
of train and validation in the regimes DeepSeek and Qwen.  Codes 3, 4, 5 take no part in training, in the choice of the
epoch or in the weights of the questions.  Rosters name every trace of the training and validation parts and of the
evaluation sets, with its original split, group and the sha256 of its canonical question key:
  main        held-out group in the original test part (its questions are outside train and validation of all solvers);
  familiar    held-out traces of the original train part whose question is in the REMAINING train part (another solver);
  unfamiliar  held-out traces of the original train part whose question is not in the remaining train part.
Held-out traces of the original validation part are in no evaluation set.
Usage: python build_regimes.py CORPUS_DIR STATUS_TABLE_CSV OUT_DIR"""
import collections as C
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FAMILY = {"gpt-oss-20b": "gpt-oss", "openai/gpt-oss-20b": "gpt-oss", "openai/gpt-oss-120b": "gpt-oss", "Qwen/Qwen3-4B": "Qwen", "Qwen/Qwen3-8B": "Qwen", "qwen_Qwen3-30B-A3B": "Qwen",
          "qwen_Qwen3-32B": "Qwen", "Qwen/Qwen3.5-122B-A10B-FP8": "Qwen", "Qwen/Qwen3.6-35B-A3B-FP8": "Qwen", "qwen3.6-35b-a3b-nitro": "Qwen", "deepseek-reasoner": "DeepSeek",
          "deepseek-v4-flash-0731-q3": "DeepSeek", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "R1-distill", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "R1-distill", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "R1-distill",
          "gemma-4-26b-a4b-it-nitro": "Gemma"}
REGIMES = [("reference", None, None)] + [(f"family_{f}", "family", f) for f in ("gpt-oss", "Qwen", "DeepSeek", "Gemma")] + [(f"benchmark_{b}", "benchmark", b) for b in ("math500", "hle", "bbh", "gpqa_diamond")]
RELATED_OUT = {"family_DeepSeek", "family_Qwen"}


def family_of(solver):
    if solver in FAMILY:
        return FAMILY[solver]
    return FAMILY[Path(solver).name]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(corpus, table, out):
    corpus, out = Path(corpus), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((corpus / "metadata.json").read_text())
    T = meta["traces"]
    status = {r["trace_id"]: r["new_status"] for r in csv.DictReader(open(table))}
    with np.load(corpus / "sequences.npz") as z:
        split, chunks = z["split"], z["chunks"]
    assert set(FAMILY) == {t["solver"] for t in T}, "the family map must name every solver label of the corpus"
    fam = np.array([family_of(t["solver"]) for t in T])
    ben = np.array([t["benchmark"] for t in T])
    grp = np.array([t["group"] for t in T])
    tsplit = np.array([{"train": 0, "validation": 1, "test": 2}[t["split"]] for t in T])
    lens = np.array([t["n_segments"] for t in T])
    off = np.concatenate([[0], np.cumsum(lens)])
    assert all(t["offset"] == off[i] for i, t in enumerate(T)) and off[-1] == len(split)
    assert np.array_equal(split, np.repeat(tsplit, lens)) and np.array_equal(chunks[:, 4], tsplit[chunks[:, 3]])          # every position and every chunk, not a sample
    qsha = np.array([hashlib.sha256(g.encode()).hexdigest() for g in grp])
    rep = dict(task_id="HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013", family_map=FAMILY, status_table_sha256=sha(table), corpus_sha256=sha(corpus / "sequences.npz"), codes=__doc__.split("Codes:")[1].split("Rosters")[0].strip(), regimes={})
    for name, kind, group in REGIMES:
        held = np.zeros(len(T), bool) if kind is None else (fam == group if kind == "family" else ben == group)
        related = (fam == "R1-distill") & (name in RELATED_OUT)
        code = tsplit.copy()
        code[held & (tsplit == 0)] = 3
        code[held & (tsplit == 1)] = 5
        code[related & (tsplit < 2)] = 4
        s2 = np.repeat(code, lens).astype(split.dtype)
        c2 = chunks.copy()
        c2[:, 4] = code[chunks[:, 3]]
        train, val = code == 0, code == 1
        assert not (held | related)[train | val].any() and not (held | related)[c2[np.isin(c2[:, 4], (0, 1)), 3]].any()
        assert not (set(grp[train]) & set(grp[tsplit == 2])) and not (set(grp[val]) & set(grp[tsplit == 2])) and not (set(grp[train]) & set(grp[val]))
        left = set(grp[train])
        main_set = held & (tsplit == 2)
        fam_set = (code == 3) & np.array([g in left for g in grp])
        unf_set = (code == 3) & ~fam_set
        p = out / f"regime_{name}.npz"
        np.savez_compressed(p, split=s2, chunks=c2)
        roster = []
        for i, t in enumerate(T):
            role = "train" if train[i] else "validation" if val[i] else "evaluation_main" if main_set[i] else "evaluation_familiar_question" if fam_set[i] else "evaluation_unfamiliar_question" if unf_set[i] else \
                "held_out_validation_not_evaluated" if code[i] == 5 else "related_excluded" if code[i] == 4 else "test_other_group"
            roster.append(dict(trace_id=t["trace_id"], role=role, original_split=t["split"], family=fam[i], solver=t["solver"], benchmark=t["benchmark"], teacher=t["teacher"], end_status=status[t["trace_id"]],
                               question_sha256=qsha[i], n_segments=int(lens[i]), offset=int(off[i])))
        rp = out / f"roster_{name}.jsonl"
        with open(rp, "w") as f:
            for r in roster:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cnt = C.Counter(r["role"] for r in roster)
        nq = lambda m: int(len(set(grp[m])))
        rep["regimes"][name] = dict(kind=kind, held_out_group=group, traces=dict(cnt), questions=dict(train=nq(train), validation=nq(val), evaluation_main=nq(main_set), evaluation_familiar_question=nq(fam_set),
                                                                                                    evaluation_unfamiliar_question=nq(unf_set)),
                                    positions=dict(train=int((s2 == 0).sum()), validation=int((s2 == 1).sum())), chunks_train=int((c2[:, 4] == 0).sum()),
                                    natural_in_main=int(sum(1 for r in roster if r["role"] == "evaluation_main" and r["end_status"] == "natural")), regime_sha256=sha(p), roster_sha256=sha(rp))
        print(name, dict(cnt), rep["regimes"][name]["questions"], "chunks", rep["regimes"][name]["chunks_train"])
    tt = C.Counter((t["teacher"], family_of(t["solver"]), t["benchmark"], t["split"]) for t in T)
    rep["teacher_source_by_family_benchmark_split"] = [dict(teacher=k[0], family=k[1], benchmark=k[2], split=k[3], traces=n) for k, n in sorted(tt.items())]
    rep["teacher_source_totals"] = dict(C.Counter(t["teacher"] for t in T))
    rep["teacher_source_by_family"] = {f"{a}|{b}": n for (a, b), n in sorted(C.Counter((t["teacher"], family_of(t["solver"])) for t in T).items())}
    (out / "REGIMES_BUILD.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(rep["teacher_source_totals"], rep["teacher_source_by_family"])
    print("REGIMES DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
