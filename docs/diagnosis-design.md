# 규칙 기반 진단 설계 초안

## 목적

진단 엔진은 `evaluation.json`에서 관찰 가능한 사실을 계산하고, 정해진 규칙에 따라 문제 후보를 구조화한다. 원인을 확정하거나 학습 설정을 자동 변경하지 않는다.

AI 챗봇은 이 결과를 사용해 자연어로 설명하지만 수치, 상태, 심각도 및 근거는 규칙 엔진의 결과를 우선한다.

## 상태의 의미

| 상태 | 의미 |
| --- | --- |
| `pass` | 이 규칙의 기준을 넘는 이상 징후가 발견되지 않음 |
| `warning` | 설정된 기준을 만족하는 문제 후보가 발견됨 |
| `insufficient_data` | 해당 규칙을 판단하기 위한 지표나 학습 기록이 부족함 |

`pass`는 모델 전체가 정상이라는 의미가 아니다. `insufficient_data`는 실패가 아니라 판단 보류다.

내부 상태 코드는 API 안정성을 위해 유지하고 사용자 화면에서는 다음 라벨을 사용한다.

| 내부 코드 | 사용자 표시 |
| --- | --- |
| `pass` | 이상 징후 없음 |
| `warning` | 확인 필요 |
| `insufficient_data` | 판단 보류 |

각 진단은 사용자용 `title`과 `status_label`을 함께 제공한다.

## 심각도와 확신도

- `severity`는 관찰된 영향의 크기다.
- `confidence`는 주어진 데이터로 규칙 판정을 신뢰할 수 있는 정도다.
- 데이터 부족 상태의 `severity`는 `none`으로 기록한다.
- 데이터가 부족하다는 사실 자체가 명확하면 `insufficient_data`의 `confidence`는 높을 수 있다.

## 1차 규칙

### 학습 기록 부족

- 3 epoch 미만: 추세 진단 불가
- 3~9 epoch: 향후 예비 진단 범위
- 10 epoch 이상: 추세 규칙 실행 가능

### Precision·Recall 불균형

- 1~2 epoch: 값이 존재해도 insufficient_data
- 3~9 epoch: 예비 판정으로 실행하고 confidence를 0.6으로 제한
- 10 epoch 이상: 정상 판정 confidence 0.9 사용
- `abs(precision - recall) >= 0.15`: warning
- `abs(precision - recall) >= 0.30`: high severity 후보
- precision 또는 recall이 없으면 insufficient_data

이 규칙은 불균형이라는 증상만 판정한다. confidence threshold, 클래스 문제 또는 작은 객체 문제를 원인으로 확정하지 않는다.

### 학습 정체

- 최소 10 epoch 필요
- 전체 epoch의 20%를 사용하되 최소 5 epoch를 한 구간으로 사용
- 마지막 두 구간의 최대 mAP50-95를 비교
- 최근 구간 최고점과 이전 구간 최고점의 차이가 0.005 미만이면 warning
- 두 구간에 누락값이 있으면 insufficient_data

### 과적합 의심

- 최소 10 epoch 필요
- 최근 전체 epoch의 20%를 사용하되 최소 5 epoch 사용
- train total loss가 10% 이상 감소
- validation total loss가 10% 이상 증가
- 같은 구간의 mAP50-95 개선 폭이 0.005 이하

세 조건이 모두 충족될 때만 warning을 생성한다. total loss는 box, cls, DFL loss 중 존재하는 값의 합으로 계산한다.

### 최적 epoch 이후 성능 하락

- 최소 10 epoch 필요
- 최고 epoch 이후 구간이 전체 기록의 20% 이상
- 최고 mAP50-95 대비 마지막 값이 0.01 이상 하락

## 전체 상태 계산

1. 하나 이상의 warning이 있으면 `overall_status`는 `warning`이다.
2. warning이 없고 하나 이상의 insufficient_data가 있으면 `insufficient_data`다.
3. 모든 규칙이 pass이면 `pass`다.

## 초안에서 의도적으로 제외한 항목

- 절대 mAP 기준의 좋음·나쁨 판단
- 클래스별 성능 및 불균형 진단
- 작은 객체 성능 진단
- 라벨 누락 및 데이터 누수 확정
- 자동 설정 변경

이 항목들은 추가 데이터 계약과 별도 승인 정책을 정의한 후 확장한다.

## 검토가 필요한 결정

1. 신뢰 가능한 추세 분석의 최소 기록을 10 epoch로 둘지
2. Precision·Recall 차이 0.15를 기본 warning 기준으로 둘지
3. 정체 기준인 mAP50-95 개선 폭 0.005가 적절한지
4. 과적합 규칙에서 세 조건을 모두 요구할지
5. `insufficient_data`를 전체 상태에 반영하는 우선순위가 적절한지
