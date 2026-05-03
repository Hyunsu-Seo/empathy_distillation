# 지식 증류 기반 sLLM 공감 대화 성능 향상 연구

> GPT-4o의 공감 대화 생성 능력을 소형 언어 모델(sLLM)에 증류하여  
> 적은 자원으로도 인간 친화적인 공감 응답을 생성하도록 학습한 연구 프로젝트입니다.

---

## 1. Project Overview

본 프로젝트는 **공감 대화 생성** 태스크에서 소형 언어 모델(sLLM)의 성능을 향상시키기 위해, 대규모 언어 모델(LLM)의 지식을 증류하는 방법을 비교한 연구입니다.

LLM은 인간과 유사한 자연어 생성 능력을 보이지만, 모델 크기와 연산 비용이 크기 때문에 실제 서비스 환경에서 활용하기에는 제약이 있습니다. 반면 sLLM은 상대적으로 적은 자원으로 구동할 수 있지만, 공감처럼 맥락 이해와 정서적 반응이 필요한 대화 생성에서는 한계를 보입니다.

이를 해결하기 위해 본 연구에서는 GPT-4o를 teacher model로 활용하여 공감 대화 증류 데이터를 구축하고, sLLM을 다음 세 가지 방식으로 학습하여 성능을 비교했습니다.

- 일반 Fine-tuning
- Chain-of-Thought(CoT) 기반 증류
- Tree-of-Thoughts(ToT) 기반 증류

최종적으로 정량 평가, G-Eval, 인간 설문조사를 통해 공감 대화 생성 태스크에서 어떤 증류 방식이 가장 효과적인지 분석했습니다.

---

## 2. Motivation

공감은 인간 대화에서 단순한 정보 전달을 넘어, 상대의 감정과 상황을 이해하고 관계를 형성하는 핵심 요소입니다. 특히 정서적 지지, 상담, 고객 응대, 헬스케어, 교육 서비스 등에서는 사용자의 감정을 이해하고 적절히 반응하는 대화 능력이 중요합니다.

하지만 일반적인 sLLM은 다음과 같은 한계를 보입니다.

- 화자의 감정을 충분히 반영하지 못함
- 상황과 관계에 맞는 말투 선택이 미흡함
- 공감 표현이 형식적이거나 단조로움
- LLM 대비 추론 능력과 문장 생성 품질이 낮음
- 공감 답변을 생성하기 위한 데이터와 학습 자원이 제한적임

본 프로젝트는 이러한 문제를 해결하기 위해, GPT-4o가 생성한 공감 응답 및 추론 정보를 활용하여 sLLM에 공감 대화 능력을 증류하는 방식을 실험했습니다.

---

## 3. Research Goals

본 연구의 주요 목표는 다음과 같습니다.

1. 공감 발화 생성 태스크를 위한 증류 데이터셋 구축
2. Fine-tuning, CoT 증류, ToT 증류 방식의 성능 비교
3. sLLM이 공감 답변을 생성할 때 필요한 추론 정보의 범위 분석
4. 정량 평가와 정성 평가를 함께 활용한 공감 응답 품질 검증
5. 저비용 공감 대화 AI 서비스 개발 가능성 탐색

---

## 4. Dataset

본 프로젝트에서는 AI Hub의 **공감형 대화** 데이터를 기반으로 실험 데이터를 구성했습니다. 이후 GPT-4o API를 활용하여 CoT 및 ToT 증류용 데이터를 구축했습니다.

| Dataset | Description |
|---|---|
| AI Hub 공감형 대화 데이터 | 기본 공감 대화 학습 데이터 |
| GPT-4o 기반 CoT 데이터 | 공감 답변 생성 근거와 최종 답변으로 구성 |
| GPT-4o 기반 ToT 데이터 | 공감 태도 후보, 태도 선택, 추론 근거, 최종 답변으로 구성 |

최종적으로 **6,000개의 공감 대화 증류 데이터**를 구축하여 학습에 사용했습니다.

---

## 5. Data Construction

### 5.1 Common Settings

GPT-4o를 활용한 데이터 생성 시 다음과 같은 공통 설정을 사용했습니다.

| Parameter | Value |
|---|---|
| Teacher Model | GPT-4o |
| 기본 Temperature | 0.7 |
| 공감 답변 생성 Temperature | 1.0 |
| 기본 역할 | You are a helpful assistant. |
| 공감 답변 역할 | 화자의 말을 듣고 적절한 공감과 반응을 하는 청자 |

공감 답변 생성 시에는 일반 assistant 역할이 아니라, 화자의 감정과 관계를 고려해 자연스럽게 반응하는 청자 역할을 부여했습니다.

---

### 5.2 CoT Dataset

CoT 데이터는 공감 답변을 생성하기 전에, 답변이 왜 적절한지에 대한 **rationale**을 함께 생성하도록 구성했습니다.

CoT 데이터 구조는 다음과 같습니다.

```text
Input: 화자의 발화
Rationale: 공감 답변을 생성해야 하는 이유
Response: 최종 공감 답변
```

CoT 방식은 ToT보다 구조가 단순하지만, 공감 답변의 근거를 함께 학습할 수 있어 모델이 화자의 감정과 상황을 고려한 응답을 생성하도록 돕습니다.

---

### 5.3 ToT Dataset

ToT 데이터는 공감 답변 생성 과정을 여러 단계로 나누어 구성했습니다.

ToT 데이터 구조는 다음과 같습니다.

```text
Input: 화자의 발화
Empathy Candidates: 공감 태도 후보 4개
Selected Empathy: 가장 적절한 공감 태도
Rationale: 답변 생성 근거
Response: 최종 공감 답변
```

ToT 방식은 공감 태도 후보 생성, 태도 선택, 근거 생성, 응답 생성까지 포함하기 때문에 CoT보다 더 넓은 범위의 추론 정보를 학습합니다.

다만 sLLM의 생성 능력과 GPU 비용을 고려하여, 학습 단계에서는 원래 ToT 구조를 완화하여 적용했습니다.

---

## 6. Models

본 프로젝트에서는 한국어 응답 생성이 가능한 sLLM 3종을 사용했습니다.

| Model | Description |
|---|---|
| EXAONE-3.5-2.4B-Instruct | LG AI Research의 한국어 특화 instruction model |
| kanana-nano-2.1b-instruct | 카카오의 소형 instruction model |
| llama-3.2-Korean-Bllossom-3B | 한국어 특화 LLaMA 계열 instruction model |

세 모델 모두 상대적으로 적은 파라미터 수를 가진 모델이며, 단일 GPU 환경에서 학습할 수 있도록 4-bit QLoRA 방식을 적용했습니다.

---

## 7. Training Methods

### 7.1 Basic Fine-tuning

Basic Fine-tuning은 AI Hub 공감형 대화 데이터의 인간 답변을 그대로 학습하는 방식입니다.

입력 구조는 다음과 같습니다.

```text
System: 당신은 화자의 발언에 공감하는 청자 역할입니다.
User: 화자의 발화
Assistant: 청자의 공감 답변
```

이 방식은 별도의 reasoning 정보 없이 화자의 발화와 공감 답변만 학습합니다. 따라서 학습 범위는 공감 답변의 표현적 특성에 한정됩니다.

---

### 7.2 CoT Distillation

CoT 증류는 공감 답변을 바로 생성하지 않고, 답변 생성 근거를 함께 학습하는 방식입니다.

CoT 학습의 loss는 다음과 같이 구성했습니다.

```text
total loss CoT = (1 - alpha) * rationale loss + alpha * response loss
```

이를 통해 모델은 공감 답변을 생성하는 능력뿐만 아니라, 그 답변이 왜 적절한지에 대한 논리적 근거도 함께 학습합니다.

---

### 7.3 ToT Distillation

ToT 증류는 공감 답변 생성 과정을 여러 하위 단계로 나누어 학습하는 방식입니다.

ToT 학습의 loss는 다음과 같이 구성했습니다.

```text
total loss ToT = alpha * (selected empathy loss + response loss)
               + (1 - alpha) * (empathy candidates loss + rationale loss)
```

ToT 방식은 공감 태도 후보 생성, 최적 태도 선택, 근거 생성, 최종 답변 생성을 함께 학습하기 때문에 가장 복합적인 증류 방식입니다.

---

## 8. Training Settings

주요 학습 설정은 다음과 같습니다.

| Parameter | Value |
|---|---|
| Epochs | 7 |
| Batch Size | 1 |
| Weight Decay | 1e-6 |
| Max Length | 400 |
| Learning Rate | 2e-5 |
| Alpha | 0.3, 0.5, 0.7, None |
| Repeated Versions | 3 |
| Fine-tuning Method | 4-bit QLoRA |

`alpha` 값은 추론 정보와 최종 답변 중 어느 쪽의 loss에 더 많은 가중치를 둘 것인지 조절하기 위해 사용했습니다.

---

## 9. Evaluation

공감 답변의 품질을 다각도로 평가하기 위해 정량 평가와 정성 평가를 함께 진행했습니다.

### 9.1 Quantitative Evaluation

정량 평가는 다음 지표를 사용했습니다.

| Metric | Description |
|---|---|
| Loss | 학습 및 검증 손실값 |
| PPL | 다음 단어 예측 시 모델의 불확실성 |
| BLEU | 생성 문장과 정답 문장의 n-gram precision |
| ROUGE | 정답 문장의 n-gram이 생성 문장에 포함되는 정도 |
| METEOR | unigram alignment 기반 문장 유사도 |

---

### 9.2 G-Eval

G-Eval은 GPT를 활용해 모델이 생성한 공감 답변을 평가하는 방식입니다.

평가 기준은 다음 세 가지로 구성했습니다.

| Criterion | Description |
|---|---|
| 문맥 이해 | 화자의 발화와 관련 있는 답변인지, 관계에 맞는 말투를 사용했는지 평가 |
| 감정 발화 | 화자의 감정을 이해하고 함께 느끼는 표현이 포함되었는지 평가 |
| 진정성 | 공감 표현이 구체적이고 감정의 깊이가 적절한지 평가 |

각 항목은 1점부터 5점까지 평가하며, 세 기준을 합산하여 최종 점수를 계산했습니다.

---

### 9.3 Human Survey

G-Eval 결과와 실제 인간 선호도가 유사한지 확인하기 위해 설문조사를 진행했습니다.

| Item | Value |
|---|---|
| Participants | 18명 |
| Gender | 남성 8명, 여성 10명 |
| Age Group | 20대 11명, 30대 3명, 40대 2명, 50대 2명 |
| Compared Models | Exaone, Kanana |
| Excluded Model | Bllossom |

Bllossom 모델은 예측 문장의 종결이 불완전한 경우가 많아 설문조사 문항 구성에서 제외했습니다.

---

## 10. Results

### 10.1 Quantitative Results

정량 평가에서는 지표별로 다른 경향이 나타났습니다.

- CoT 증류는 Loss와 PPL 측면에서 가장 안정적으로 학습되는 경향을 보였습니다.
- Basic fine-tuning은 BLEU, ROUGE, METEOR와 같은 정답 문장 유사도 기반 지표에서 강점을 보였습니다.
- ToT 증류는 CoT보다 복잡한 구조를 학습하기 때문에 Loss와 PPL이 높게 나타났으나, 일부 n-gram 유사도 지표에서는 CoT보다 높은 성능을 보였습니다.

이는 정답 문장과의 표면적 유사도가 높다고 해서 반드시 인간이 더 공감적으로 느끼는 답변을 생성한다는 의미는 아님을 보여줍니다.

---

### 10.2 G-Eval Results

G-Eval 기준에서는 세 모델 모두 공통적으로 다음 순서를 보였습니다.

```text
CoT Distillation > ToT Distillation > Basic Fine-tuning
```

특히 Kanana 모델이 전체적으로 가장 높은 평가 점수를 보였고, 그다음으로 EXAONE, Bllossom 순서의 결과를 보였습니다.

또한 세 평가 기준에서는 다음과 같은 경향이 나타났습니다.

```text
문맥 이해 점수 > 감정 발화 점수 > 진정성 점수
```

이는 sLLM이 화자의 상황과 관계를 이해하는 능력은 증류를 통해 비교적 잘 향상되었지만, 구체적이고 깊이 있는 진정성 표현 생성에는 여전히 한계가 있음을 의미합니다.

---

### 10.3 Human Survey Results

설문조사 결과도 G-Eval과 유사한 경향을 보였습니다.

| Method | 1·2순위 선택 비율 |
|---|---:|
| CoT Distillation | 91.67% |
| ToT Distillation | 66.67% |
| Basic Fine-tuning | 41.67% |

즉, 인간 평가에서도 CoT 증류 방식으로 학습된 모델의 답변이 가장 선호되었습니다.

설문조사 이유 선택 결과에서는 사람들이 공감 답변을 평가할 때 다음 순서의 기준을 중요하게 보는 것으로 나타났습니다.

```text
문맥 이해 > 감정 발화 > 진정성
```

---

## 11. Key Findings

본 프로젝트의 주요 발견은 다음과 같습니다.

1. sLLM에 공감 대화 능력을 증류할 때는 과도하게 많은 추론 정보를 주입하는 것보다, 답변 생성에 필요한 핵심 근거를 함께 학습시키는 CoT 방식이 효과적이었습니다.

2. ToT 방식은 공감 태도 후보, 태도 선택, 근거 생성, 응답 생성을 모두 학습하기 때문에 정보량은 많지만, sLLM 입장에서는 학습 부담이 커져 성능이 안정적으로 향상되지 않았습니다.

3. Basic fine-tuning은 정답 문장과의 표면적 유사도는 높았지만, G-Eval과 인간 평가에서는 CoT 증류보다 낮은 선호도를 보였습니다.

4. 공감 응답 품질은 단순히 BLEU, ROUGE, METEOR 같은 정량 지표만으로 판단하기 어렵고, G-Eval 및 인간 평가와 함께 해석해야 합니다.

5. 모델 크기가 크다고 항상 더 좋은 결과를 보이는 것은 아니며, 6,000개라는 제한된 데이터에서는 kanana-nano-2.1b-instruct가 가장 높은 정성 평가 점수를 보였습니다.
