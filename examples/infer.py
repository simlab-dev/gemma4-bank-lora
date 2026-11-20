"""Gemma-4 금융(은행) 상담 LoRA 어댑터 추론 예제 (transformers + PEFT).

    python examples/infer.py --adapter adapters/E2B/topic --task topic         --question "해외여행 가기 전에 달러로 바꾸려면 어떻게 해야 하나요?"

과제별로 필요한 입력:
    topic, situation, terms, answer   --question (고객 문의)
    followup                          --question (고객 문의) --answer (상담원 답변)
    summary                           --transcript 또는 --transcript-file
                                      (한 줄에 발화 1개, 상담사 "TX " / 고객 "RX ")

평가 때와 같은 조건으로 생성한다: 사용자 메시지 1개 · 시스템 메시지 없음 ·
채팅 템플릿(add_generation_prompt=True) · greedy · 반복 페널티 없음.
"""
import argparse
import json
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 과제별 생성 상한 (토큰) — docs/사용가이드.md §4
MAX_NEW_TOKENS = {
    "topic": 32, "situation": 32, "terms": 32,
    "followup": 384, "summary": 512, "answer": 1024,
}
INPUT_KEYS = ("question", "answer", "transcript")


def build_prompt(task, inputs):
    """프롬프트 틀의 {자리표시자} 만 바꾼다 (str.format 은 쓰지 않는다)."""
    text = (ROOT / "prompts" / f"{task}.txt").read_text(encoding="utf-8").rstrip("\n")
    for k in INPUT_KEYS:
        if inputs.get(k) is not None:
            text = text.replace("{" + k + "}", inputs[k])
    if "{" in text and any("{" + k + "}" in text for k in INPUT_KEYS):
        raise SystemExit(f"[{task}] 입력이 빠졌습니다: {text}")
    return text


def load(adapter_dir):
    cfg = json.loads((pathlib.Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    base, rev = cfg["base_model_name_or_path"], cfg.get("revision")
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=rev, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def stop_ids(tok, model):
    """정지 토큰 = generation_config eos ∪ 토크나이저 eos ∪ 턴 종료 토큰 <turn|>.

    generate(eos_token_id=...) 는 generation_config 값을 대체하므로 모두 합쳐 넘긴다.
    """
    ids = set()
    for v in (model.generation_config.eos_token_id, tok.eos_token_id):
        if isinstance(v, int):
            ids.add(v)
        elif v:
            ids.update(v)
    turn_end = tok.convert_tokens_to_ids("<turn|>")
    if isinstance(turn_end, int) and turn_end != tok.unk_token_id:
        ids.add(turn_end)
    return sorted(ids)


@torch.inference_mode()
def generate(tok, model, task, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    stops = stop_ids(tok, model)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS[task], do_sample=False,
                         eos_token_id=stops or None)
    new = out[0, enc["input_ids"].shape[1]:]
    finished = any(int(t) in stops for t in new.tolist())
    return tok.decode(new, skip_special_tokens=True).strip(), finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="예: adapters/E2B/topic")
    ap.add_argument("--task", required=True, choices=sorted(MAX_NEW_TOKENS))
    ap.add_argument("--question")
    ap.add_argument("--answer", help="followup 과제의 상담원 답변")
    ap.add_argument("--transcript")
    ap.add_argument("--transcript-file")
    a = ap.parse_args()

    transcript = a.transcript
    if a.transcript_file:
        transcript = pathlib.Path(a.transcript_file).read_text(encoding="utf-8").rstrip("\n")
    prompt = build_prompt(a.task, {"question": a.question, "answer": a.answer,
                                   "transcript": transcript})

    tok, model = load(a.adapter)
    answer, finished = generate(tok, model, a.task, prompt)
    print(answer)
    if not finished:
        # 정지 토큰 없이 상한에 닿은 답 = 반복 루프일 가능성이 높다 → 버리거나 다시 시도
        print("[경고] 생성 상한에 닿았습니다. 반복 루프일 수 있습니다.")


if __name__ == "__main__":
    main()
