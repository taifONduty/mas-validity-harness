"""Outcome scoring. This is the ground truth column of the whole study.

math : gold answers ship in the record ('#### 3360'); we parse the model's
       'FINAL ANSWER:' line (fallback: last number in the text).
code : records ship their own `check(candidate)` unit tests; we execute
       candidate + tests in a subprocess with a timeout.

WARNING: executing model-generated code runs arbitrary code. Do it inside a
VM/container you don't care about, never on your main machine unprotected.
"""
import json
import re
import subprocess
import sys
import tempfile


# ------------------------------ math ------------------------------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text):
    m = _NUM.findall(text.replace(",", ""))
    return m[-1] if m else None


def parse_gold_math(answer_field):
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    return m.group(1).replace(",", "").strip() if m else None


_FA = re.compile(r"FINAL ANSWER\s*[:\-]?\s*[\*\_\$\s]*(-?\d[\d,]*(?:\.\d+)?)", re.I)


def parse_pred_math(final_text):
    r"""Take the LAST 'FINAL ANSWER: <number>' occurrence; tolerate markdown
    bold/underscore/currency between the colon and the number; the capture
    group REQUIRES a digit (the old class [\d,\.]+ could match bare dots).
    Fallback: last number anywhere in the text."""
    m = _FA.findall(final_text)
    if m:
        return m[-1].replace(",", "").strip()
    return _last_number(final_text)


def score_math(traj):
    gold = parse_gold_math(traj["record_meta"].get("answer", ""))
    pred = parse_pred_math(traj["final_answer"])
    if gold is None or pred is None:
        return 0
    try:
        return int(abs(float(gold) - float(pred)) < 1e-6)
    except ValueError:
        return int(gold == pred)


# ------------------------------ code ------------------------------
# Faithful replica of the benchmark's own evaluation pipeline
# (lost_in_conversation/tasks/code/task_code.py::evaluator_function), using
# the repo's vendored LiveCodeBench evaluator (tasks/code/eval_code.py).
# Key protocol detail we originally missed: shards almost never reveal the
# canonical function name (5/45 HumanEval, 0/55 LCB), so the OFFICIAL
# protocol RENAMES the model's first defined function to metadata.func_name
# before testing. Unscorable records get outcome None (excluded), never a
# silent 0.
import ast as _ast
import base64 as _b64
import importlib.util as _ilu
import os as _os
import pickle as _pickle
import zlib as _zlib

_EVAL_MOD = None


def extract_code_block(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text  # fall back to raw text


def _load_eval_module(repo_path="lost_in_conversation"):
    global _EVAL_MOD
    if _EVAL_MOD is None:
        p = _os.path.join(repo_path, "tasks", "code", "eval_code.py")
        spec = _ilu.spec_from_file_location("lic_eval_code", p)
        _EVAL_MOD = _ilu.module_from_spec(spec)
        spec.loader.exec_module(_EVAL_MOD)
    return _EVAL_MOD


def _load_test_cases(rec):
    """Replica of TaskCode.load_test_cases: public + (possibly compressed)
    private tests -> the JSON blob run_test expects."""
    public = json.loads(rec["public_test_cases"])
    private = []
    if "private_test_cases" in rec:
        try:
            private = json.loads(rec["private_test_cases"])
        except Exception:
            private = json.loads(_pickle.loads(_zlib.decompress(
                _b64.b64decode(rec["private_test_cases"].encode("utf-8")))))
    return json.dumps({
        "inputs": [t["input"] for t in public + private],
        "outputs": [t["output"] for t in public + private],
        "fn_name": rec["metadata"].get("func_name", None),
    })


def score_code(traj, source_rec, repo_path="lost_in_conversation", timeout=6):
    """Returns 1 (all tests pass), 0 (fails), or None (unscorable record)."""
    if source_rec is None or "public_test_cases" not in source_rec:
        return None
    code = extract_code_block(traj["final_answer"])
    code = code.replace("```python", "").replace("```", "")
    if "def " not in code:
        return 0                       # their protocol: no function = fail

    # HumanEval: prepend the prompt's imports (their exact step)
    if "prompt" in source_rec:
        try:
            imports = [_ast.unparse(n) for n in _ast.parse(source_rec["prompt"]).body
                       if isinstance(n, (_ast.Import, _ast.ImportFrom))]
            if imports:
                code = "\n".join(imports) + "\n\n" + code
        except SyntaxError:
            pass

    # Their exact rename: first defined name -> canonical func_name
    fn = source_rec["metadata"]["func_name"]
    old_name = code.split("def ")[1].split("(")[0].strip()
    code = code.replace(old_name, fn)

    tests = _load_test_cases(source_rec)
    ev = _load_eval_module(repo_path)
    try:
        outputs, _meta = ev.check_correctness(source_rec, code, tests, timeout=timeout)
        return int(all(o is True for o in outputs))
    except Exception:
        return 0

def score_file(traj_path, out_path, data_path="data_sharded.json",
               repo_path="lost_in_conversation"):
    """Read trajectories JSONL, append 'outcome', write scored JSONL.
    Code scoring joins back to the source dataset by task_id."""
    records = {r["task_id"]: r for r in json.load(open(data_path))}
    n, ok, unscorable = 0, 0, 0
    by_source = {}
    with open(traj_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            traj = json.loads(line)
            if traj["task"] == "math":
                traj["outcome"] = score_math(traj)
            elif traj["task"] == "code":
                traj["outcome"] = score_code(traj, records.get(traj["task_id"]),
                                             repo_path=repo_path)
            else:
                traj["outcome"] = None
            n += 1
            if traj["outcome"] is None:
                unscorable += 1
            else:
                ok += traj["outcome"]
                src = traj["task_id"].split("/")[0]
                a, b = by_source.get(src, (0, 0))
                by_source[src] = (a + traj["outcome"], b + 1)
            fout.write(json.dumps(traj) + "\n")
    print(f"scored {n} trajectories, success rate = {ok}/{n-unscorable} = "
          f"{ok/max(n-unscorable,1):.2f} | unscorable: {unscorable}")
    for src, (a, b) in sorted(by_source.items()):
        print(f"  {src}: {a}/{b} = {a/b:.2f}")


if __name__ == "__main__":
    score_file(*sys.argv[1:])
