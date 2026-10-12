# 과적합 의심 진단 테스트 사례 초안

## 목적

`suspected_overfitting` 규칙을 구현하기 전에 손실 계산 방식, 비교 구간, 경계값 및 예상 판정을 합의한다. 이 규칙은 과적합을 확정하지 않고 여러 지표가 동시에 보이는 경우에만 문제 후보를 제시한다.

## 계산 방식 제안

### 사전 조건

- 최소 학습 기록: 10 epoch
- 구간 크기: `max(5, ceil(전체 epoch 수 × 0.20))`
- 비교 구간: 마지막 두 구간
- 필요한 지표: train 및 validation의 box loss, cls loss, DFL loss, validation mAP50-95
- 최근 두 구간에 필요한 값이 하나라도 없으면 `insufficient_data`

### Epoch별 total loss

```text
train total loss = train box loss + train cls loss + train DFL loss
validation total loss = validation box loss + validation cls loss + validation DFL loss
```

세 loss는 서로 다른 목적과 스케일을 가질 수 있으므로 total loss 자체의 절대값으로 모델 간 성능을 비교하지 않는다. 동일 학습 실행 안에서 시간에 따른 방향만 비교한다.

### 구간별 변화율

```text
이전 train 평균 = 이전 구간 train total loss 평균
최근 train 평균 = 최근 구간 train total loss 평균
train 감소율 = (이전 train 평균 - 최근 train 평균) / 이전 train 평균

이전 validation 평균 = 이전 구간 validation total loss 평균
최근 validation 평균 = 최근 구간 validation total loss 평균
validation 증가율 = (최근 validation 평균 - 이전 validation 평균) / 이전 validation 평균

mAP 개선 폭 = 최근 구간 최고 mAP50-95 - 이전 구간 최고 mAP50-95
```

이전 구간 평균 loss가 0 이하이면 변화율을 안전하게 계산할 수 없으므로 판단을 보류한다.

### 과적합 의심 조건

아래 조건을 모두 만족할 때만 warning으로 판정한다.

```text
train loss 감소율 >= 0.10
validation loss 증가율 >= 0.10
mAP50-95 개선 폭 <= 0.005
```

한 조건이라도 만족하지 않으면 `pass`다. 이는 과적합이 없다는 확정이 아니라 이 규칙에 필요한 복합 증거가 부족하다는 의미다.

---

## 사례 1. 학습 기록 부족

### 입력

```text
전체 epoch: 8
```

### 예상 결과

```text
title: 과적합 의심
status: insufficient_data
status_label: 판단 보류
severity: none
confidence: 1.0
```

### 사용자 설명

> 학습 기록이 8 epoch로 최소 기준인 10 epoch보다 짧아 train·validation 추세를 비교할 수 없습니다.

---

## 사례 2. 정상적인 동반 개선

### 입력 요약

```text
이전 train 평균 loss: 3.00
최근 train 평균 loss: 2.55
train 감소율: 15%

이전 validation 평균 loss: 3.20
최근 validation 평균 loss: 2.90
validation 증가율: -9.375%

이전 최고 mAP50-95: 0.40
최근 최고 mAP50-95: 0.44
mAP 개선 폭: 0.04
```

### 예상 결과

```text
status: pass
status_label: 이상 징후 없음
severity: none
confidence: 0.9
```

### 이유

Train loss뿐 아니라 validation loss와 mAP도 함께 개선되고 있으므로 과적합 복합 조건을 만족하지 않는다.

---

## 사례 3. 과적합 의심

### 입력 요약

```text
이전 train 평균 loss: 3.00
최근 train 평균 loss: 2.40
train 감소율: 20%

이전 validation 평균 loss: 3.00
최근 validation 평균 loss: 3.45
validation 증가율: 15%

이전 최고 mAP50-95: 0.500
최근 최고 mAP50-95: 0.502
mAP 개선 폭: 0.002
```

### 예상 결과

```text
status: warning
status_label: 확인 필요
severity: medium
confidence: 0.9
```

### 사용자 설명

> Train loss는 20% 감소했지만 validation loss는 15% 증가했고 mAP50-95 개선은 0.002에 그쳤습니다. 과적합 가능성을 확인해야 합니다.

### 원인 후보

- 학습 데이터의 양이나 다양성 부족
- augmentation 또는 정규화 부족
- 현재 데이터 규모에 비해 모델 용량이 큼
- 필요 이상으로 긴 학습

### 권장 확인

- 최고 성능 epoch 이후 validation loss와 mAP 변화 확인
- train·validation 데이터 분포 차이 확인
- augmentation, weight decay 및 early stopping 설정 검토

---

## 사례 4. 일부 조건만 만족

### 입력 요약

```text
train 감소율: 14%
validation 증가율: 6%
mAP 개선 폭: 0.003
```

### 예상 결과

```text
status: pass
status_label: 이상 징후 없음
```

### 이유

Validation loss 증가율이 10% 기준에 미달한다. Train loss 감소와 mAP 정체만으로 과적합이라고 판단하지 않는다.

---

## 사례 5. 정확히 경계값

### 입력 요약

```text
train 감소율: 0.10
validation 증가율: 0.10
mAP 개선 폭: 0.005
```

### 예상 결과

```text
status: warning
status_label: 확인 필요
severity: medium
```

### 이유

감소율과 증가율은 `이상`, mAP 개선 폭은 `이하` 조건이므로 세 경계값을 정확히 만족하면 warning이다. 구현 시 부동소수점 오차를 고려한다.

---

## 사례 6. 필요한 loss 누락

### 입력 예시

```text
최근 비교 구간 중 일부 epoch의 validation DFL loss: null
```

### 예상 결과

```text
status: insufficient_data
status_label: 판단 보류
severity: none
confidence: 1.0
```

### 사용자 설명

> 최근 비교 구간의 validation DFL loss가 누락되어 과적합 여부를 안정적으로 판단할 수 없습니다.

누락된 loss를 0으로 대체하거나 남은 loss만 더하지 않는다.

---

## 사례 7. 변화율 계산 불가

### 입력 예시

```text
이전 구간 train 평균 total loss: 0
```

### 예상 결과

```text
status: insufficient_data
status_label: 판단 보류
```

### 이유

변화율 계산의 분모가 0이므로 임의의 큰 값이나 0%로 처리하지 않는다.

---

## 사례 8. mAP가 실제로 하락

### 입력 요약

```text
train 감소율: 18%
validation 증가율: 13%
mAP 개선 폭: -0.02
```

### 예상 결과

```text
status: warning
status_label: 확인 필요
severity: medium
```

### 설명 원칙

음수인 mAP 개선 폭은 `0.005 이하` 조건을 만족한다. 하락 시점과 크기에 대한 상세 설명은 향후 `post_peak_degradation` 규칙이 담당한다.

## 중복 진단 처리

같은 실행에서 `training_plateau`, `suspected_overfitting`, `post_peak_degradation`이 동시에 warning일 수 있다. 첫 버전은 결과를 임의로 하나로 합치지 않는다. 챗봇 계층에서 다음처럼 관계를 설명한다.

> 최근 성능 개선이 정체됐으며, train loss는 계속 감소하지만 validation loss는 증가해 과적합 가능성이 함께 관찰됩니다.

## 구현 전 승인할 기준

1. Epoch별 세 loss의 합을 total loss로 사용한다.
2. 최근 두 구간의 total loss 평균을 비교한다.
3. Train 감소 10%, validation 증가 10%, mAP 개선 0.005 이하를 모두 요구한다.
4. 필요한 값이 하나라도 누락되면 판단을 보류한다.
5. 경계값을 정확히 만족하는 경우 warning으로 포함한다.
6. 첫 버전의 warning 심각도는 `medium`으로 고정한다.
