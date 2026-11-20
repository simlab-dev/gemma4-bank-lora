# Gemma-4 금융(은행) 상담 LoRA 어댑터

은행 고객상담 데이터로 학습한 **Gemma-4 E2B · E4B LoRA 어댑터 12개** 입니다 (6과제 × 2크기).

LoRA adapters for Gemma-4 E2B / E4B, fine-tuned on Korean bank customer-service consultations: topic / situation / key-term classification, summarization, follow-up question and answer generation.

- 원본 모델: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) · [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) (Apache 2.0)
- 학습 데이터: AI Hub 「금융분야 고객상담 데이터」 은행 분야 ([출처](#출처--라이선스))
- 학습 방식: LoRA r=16 · α=32 · attention 층(q/k/v/o)만 · 학습률 2e-4 · 유효 배치 16

> 이 저장소에는 **어댑터 가중치 · 프롬프트 틀 · 라벨 목록 · 사용법 · 결과 요약** 만 있습니다. 학습 데이터 원문은 들어 있지 않습니다.

---

## 과제와 성능

| 과제 | 입력 → 출력 | 지표 | 원본 E2B → **학습 E2B** | 원본 E4B → **학습 E4B** | 판단 |
|---|---|---|---|---|---|
| `topic` 주제분류 (10종) | 고객 문의 → 상담 주제 | 정확도 | 51.8% → **77.0%** | 58.6% → **77.8%** | ✅ 학습 효과 큼 |
| `situation` 상황분류 (3종) | 고객 문의 → 상담 상황 | macro F1 | 0.397 → **0.497** | 0.465 → **0.518** | △ 효과 있으나 낮음 |
| `terms` 용어추출 (269종) | 고객 문의 → 핵심 금융 용어 1개 | macro F1 | 0.264 → **0.660** | 0.400 → **0.659** | ✅ 학습 효과 큼 |
| `summary` 요약 | 상담 전문 → 요약 | 원본 대비 승률 | **100%** | **99%** | ✅ 학습본 우세 (길이 효과 포함) |
| `followup` 후속질문 | 문의 + 답변 → 고객의 다음 질문 | 원본 대비 승률 | 39.5% | 39.0% | ❌ **원본보다 못함** |
| `answer` 답변생성 | 고객 문의 → 상담원 답변 | 원본 대비 승률 | 51.5% | 44.5% | ❌ 원본과 차이 없음 |

- 분류는 같은 평가 문항 2,000건, 생성은 100건을 **블라인드 쌍 판정** (원본·학습본 답의 이름과 순서를 가리고 고정 기준으로 승/패/무 판정, 50% = 동등) 한 결과입니다.
- **실사용 권장: `topic` · `terms` · `summary`.** `situation` 은 효과가 작고, `followup` · `answer` 는 원본보다 낫지 않습니다 (실험 기록으로 공개). 이유는 [docs/결과요약.md](docs/결과요약.md) 에 있습니다.
- E4B 는 E2B 보다 크지만 학습 후 성능은 같습니다 (`situation` 만 근소 우세). 12B · 26B-A4B · 31B 도 학습했지만 결과가 같아 어댑터는 공개하지 않았습니다. 다섯 크기 전체 결과는 [docs/결과요약.md](docs/결과요약.md) 에 있습니다.

## 빠른 시작

```bash
pip install "transformers>=5.8,<5.15" "peft>=0.21" accelerate torch
```

> Gemma-4 는 transformers 5.15 이상에서 로드 오류가 납니다.

```bash
# 주제분류
python examples/infer.py --adapter adapters/E2B/topic --task topic \
  --question "해외여행 가기 전에 달러로 바꾸려면 어떻게 해야 하나요?"

# 요약 (상담 전문은 한 줄에 발화 1개, 상담사 "TX " / 고객 "RX " 로 시작)
python examples/infer.py --adapter adapters/E2B/summary --task summary --transcript-file call.txt
```

원본 모델은 `adapter_config.json` 에 적힌 Hugging Face 저장소·revision 에서 자동으로 받습니다.

**입력 형식이 학습 때와 다르면 성능이 떨어집니다.** 과제별 프롬프트 틀, 생성 설정, 출력 처리는 [docs/사용가이드.md](docs/사용가이드.md) 를 보세요.

## 저장소 구성

```
adapters/E2B/<과제>/     E2B 어댑터 6개 (각 21.5MB)
adapters/E4B/<과제>/     E4B 어댑터 6개 (각 36.3MB)
prompts/<과제>.txt       과제별 프롬프트 틀 (학습 때 쓴 원문 그대로)
labels/<과제>.labels.txt 분류 과제 라벨 목록 (topic · situation · terms)
docs/사용가이드.md       입력 형식 · 생성 설정 · 출력 처리 · 주의점
docs/결과요약.md         실험 방법 · 5개 크기 전체 결과
examples/infer.py        transformers + PEFT 추론 예제
SHA256SUMS.txt           어댑터 파일 해시
```

각 어댑터 폴더에는 `adapter_config.json` 과 `adapter_model.safetensors` 만 있습니다.

## 알려진 한계

- **`answer` 는 조회 결과·금리·기간·서류를 단정하거나 지어냅니다.** 학습 정답이 실제 계좌 조회를 전제로 한 답이 많기 때문입니다. 고객에게 바로 내보내면 안 됩니다.
- **`followup` 은 답변에서 이미 안내한 내용을 다시 묻는 질문을 자주 만듭니다** (학습 데이터의 특성).
- **`situation` 은 질문 한 문장만으로 상황이 정해지지 않아** 상한이 낮습니다 (소수 클래스 '민원응대' 를 잘 찾지 못함).
- **`summary` 의 승률에는 길이 효과가 섞여 있습니다.** 원본은 길이 지시 없이 길게 써서 감점을 받았습니다. 학습본이 가려진 값을 지어낸 사례가 드물게 있습니다.
- 학습 데이터의 개인정보·민감 값은 `[마스킹]` · `[금액]` · `[숫자]` 로 바뀌어 있습니다. 실제 입력에는 진짜 값이 들어오므로 분포가 다릅니다.
- 생성 과제 판정은 LLM 1명이 고정 기준으로 블라인드 판정했고, 평가셋이 100건입니다. 사람 검수는 하지 않았습니다. 모든 결과는 단일 시드 · 1회 학습 값입니다.

## 출처 · 라이선스

**데이터 출처**

> 이 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "금융분야 고객상담 데이터"를 활용하여 학습되었습니다.
> (데이터: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71926)

학습 데이터 원문은 이 저장소에 포함되어 있지 않습니다. 데이터가 필요하면 AI Hub 에서 직접 신청하세요.

**라이선스**

- 이 저장소(어댑터 가중치 · 코드 · 문서): [MIT](LICENSE)
- 원본 모델 Gemma 4: [Apache 2.0](LICENSE-BASE-MODEL) — 어댑터를 쓰려면 원본 모델이 필요하며, 원본 모델의 라이선스 조건은 그대로 적용됩니다.
- 고지 사항: [NOTICE](NOTICE)

## 무결성 확인

```bash
sha256sum -c SHA256SUMS.txt
```
