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
def extract_code_block(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text  # fall back to raw text


def score_code(traj, timeout=15):
    candidate = extract_code_block(traj["final_answer"])
    test = traj["record_meta"].get("test", "")
    prompt = traj["record_meta"].get("prompt", "")
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    entry = m.group(1) if m else None
    if not test or not entry:
        return 0
    program = f"{candidate}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return int(r.returncode == 0)
    except subprocess.TimeoutExpired:
        return 0


SCORERS = {"math": score_math, "code": score_code}


def score_file(traj_path, out_path):
    """Read trajectories JSONL, append 'outcome' field, write scored JSONL."""
    n, ok = 0, 0
    with open(traj_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            traj = json.loads(line)
            scorer = SCORERS.get(traj["task"])
            traj["outcome"] = scorer(traj) if scorer else None
            n += 1
            ok += traj["outcome"] or 0
            fout.write(json.dumps(traj) + "\n")
    print(f"scored {n} trajectories, success rate = {ok}/{n} = {ok/max(n,1):.2f}")


if __name__ == "__main__":
    score_file(sys.argv[1], sys.argv[2])
