# Final Project Execution Plan

안녕하세요, 조교님 피드백 반영해서 final project 진행 방향을 정리했습니다.

큰 모델/H100 scale-up보다는, 남은 시간 안에 현재 구축한 `Qwen3-0.6B + EAGLE3 + SpecForge/SGLang` 환경에서 실험을 끝까지 닫는 방향으로 가려고 합니다. MetaSD에서 영감을 받되, 우리는 최근 speculative decoding에서 많이 쓰이는 EAGLE3 기반 drafter를 직접 domain별로 학습하고, 여러 RL/bandit selection policy를 비교하는 쪽으로 정리하면 될 것 같습니다. 

## Overall Goal

핵심 목표는 다음을 보이는 것입니다.

1. 우리가 직접 훈련한 EAGLE3 drafters가 domain별로 average accept length 차이를 보인다.
2. 이 차이를 이용해 RL/bandit selection이 random 또는 fixed drafter보다 더 좋은 선택을 할 수 있다.
3. 시간이 허용되면 per-request selection을 넘어 per-turn/per-speculative-round selection까지 확장한다.

## Experimental Setup

Fair comparison을 위해 train/test split은 아래처럼 맞춥니다.

Training:

- `gsm8k`: `N` samples
- `mmlu`: `N` samples
- `sharegpt`: `N` samples

Testing:

- training set과 겹치지 않는 `gsm8k`: `M` samples
- training set과 겹치지 않는 `mmlu`: `M` samples
- training set과 겹치지 않는 `sharegpt`: `M` samples

Test sequence:

- 최종 selection 실험에서는 `gsm8k M + mmlu M + sharegpt M`을 random mix한 test sequence를 사용합니다.
- 제가 training에 쓸 dataset뿐 아니라 test에 쓸 mixed dataset까지 같이 공유드리겠습니다.

Primary metric:

- `average accept length = generated output tokens / target verification steps`
- draft tokens는 모든 실험에서 `4`로 고정합니다.

## Experiment 1: Domain-Specialized EAGLE3 Drafters

우선 제가 아래 EAGLE3 drafters를 훈련하고, domain-by-drafter accept length matrix를 만들겠습니다.

- `eagle3-gsm8k`
- `eagle3-mmlu`
- `eagle3-sharegpt`
- 가능하면 `eagle3-all`

Eval domains:

- GSM8K eval
- MMLU eval
- ShareGPT eval

Expected output:

| Eval domain | `eagle3-gsm8k` | `eagle3-mmlu` | `eagle3-sharegpt` | `eagle3-all` |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | TBD | TBD | TBD | TBD |
| MMLU | TBD | TBD | TBD | TBD |
| ShareGPT | TBD | TBD | TBD | TBD |

이 matrix가 이번 프로젝트의 가장 중요한 baseline입니다. Matching domain에서 specialized drafter의 accept length가 더 높게 나오는지 확인하는 것이 1차 목표입니다.

## Experiment 2: Per-Request Selection

Per-request selection은 각 request마다 하나의 drafter를 선택하는 방식입니다.

구현은 우선 offline simulation으로 진행합니다.

1. 각 test request에 대해 모든 drafter의 accept length를 미리 측정합니다.
2. 이 accept length table을 reward table로 사용합니다.
3. RL/bandit policy가 mixed test sequence를 따라가며 request마다 하나의 drafter를 선택합니다.
4. 선택한 drafter의 accept length를 reward로 받고 policy를 update합니다.

Baseline/oracle:

- random selection
- best single drafter (아마 eagle3-all 일 것)
- domain oracle (domain label을 알고 해당 drafter 선택하는)

## Experiment 3: Per-Turn / Per-Speculative-Round Selection

Per-turn 또는 per-speculative-round selection은 speculative decoding step마다 drafter를 바꾸는 방식입니다.

이 방식은 SGLang runtime 수정이 필요할 가능성이 있어서, per-request selection보다 risk가 큽니다. 따라서 main result는 per-request selection으로 두고, per-turn/per-speculative-round selection은 가능하면 추가 결과로 넣는 방향이 현실적일 것 같습니다.

## Work Distribution

### 이도규

월요일 오후까지:

- `eagle3-gsm8k`, `eagle3-mmlu`, `eagle3-sharegpt` 훈련
- 가능하면 `eagle3-all`까지 훈련
- domain별 eval set에서 average accept length 측정
- domain-by-drafter accept length matrix 준비
- weight, train/test dataset, eval result, 실행 코드 공유

월요일 저녁-화요일 밤:

- per-turn/per-speculative-round selection 구현 시도

목요일 오후:

- 최종 포스터 인쇄

### 팀원 1

월요일 저녁-화요일 밤:

- per-request selection 실험 담당 (1)
- baseline으로서의 random selection 실험
- `epsilon-greedy` 구현 및 실험
- `UCB1` 구현 및 실험
- 더 좋아 보이는 알고리즘이 있으면 추가 실험해도 좋습니다.

Input:

- 제가 공유한 mixed test sequence
- 각 request별 drafter accept length reward table

Output:

- algorithm별 평균 accpetance length
- 간단한 observation summary

### 팀원 2

월요일 저녁-화요일 밤:

- per-request selection 실험 담당 (2)
- Oracle으로서의 domain oracle 실험
- `Thompson sampling` 구현 및 실험
- `EXP3` 구현 및 실험
- 더 좋아 보이는 알고리즘이 있으면 추가 실험해도 좋습니다.

Input:

- 제가 공유한 mixed test sequence
- 각 request별 drafter accept length reward table

Output:

- algorithm별 평균 accpetance length
- 간단한 observation summary

### 팀원 3

화요일 밤-수요일 밤:

- 팀원 1, 2가 구현한 algorithms를 per-turn/per-speculative-round selection 쪽에 적용
- 이도규가 per-speculative-round selection이 가능한 SGLang 수정 코드를 제공하면, 해당 runtime에 A1-A4 policy를 연결
- 구현이 어려우면 per-request 결과 분석 또는 ablation 보강

### 팀원 4

화요일 밤-수요일 낮:

- 우선 per-request selection 결과까지 반영해서 포스터 초안 제작

수요일 밤-목요일 점심:

- 팀원 3의 per-turn/per-speculative-round 결과가 나오면 추가 반영
- 목요일 점심까지 최종 포스터 PDF 완성 목표

## Schedule

| Time | Goal |
| --- | --- |
| 월요일 오후까지 | EAGLE3 drafter training, eval dataset, reward table, code 공유 |
| 월요일 저녁-화요일 밤 | per-request algorithms 실험, per-turn runtime 구현 시도 |
| 화요일 밤-수요일 밤 | per-turn selection 적용 또는 per-request 분석 보강 |
| 화요일 밤-수요일 낮 | 포스터 초안 제작 |
| 수요일 밤-목요일 점심 | 추가 결과 반영 및 최종 포스터 PDF 완성 |
| 목요일 오후 | 포스터 인쇄 |

## Framing Against MetaSD

우리 프로젝트는 MetaSD에서 영감을 받지만, 완전히 새로운 bandit algorithm을 제안하는 것이 주 목표는 아닙니다.

차별점은 다음처럼 잡는 것이 좋겠습니다.

- MetaSD-style adaptive drafter selection을 EAGLE3 + SpecForge + SGLang 기반으로 구현한다.
- 기존 drafter pool을 가정하지 않고, 동일 target model에 대해 domain-specialized EAGLE3 drafters를 직접 훈련한다.
- Block Divergence 대신 serving stack에서 직접 관찰 가능한 average accept length를 reward로 사용한다.
- Per-request selection을 main result로 두고, 가능하면 per-speculative-round selection까지 확장한다.
