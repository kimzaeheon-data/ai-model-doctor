# YOLO 결과 업로드 및 표준 입력 계약

## 목적

사용자가 업로드한 Ultralytics YOLO 객체 탐지 학습 결과를 Parser가 하나의 표준 `evaluation.json`으로 변환한다. 진단 엔진과 챗봇은 YOLO의 원본 파일 구조에 직접 의존하지 않고 이 JSON만 사용한다.

## 지원 범위

- 첫 지원 프레임워크: Ultralytics YOLO
- 첫 지원 작업: 객체 탐지(`detect`)
- 스키마 버전: `0.1.0`
- 업로드 방식: YOLO 학습 결과 디렉터리를 압축한 ZIP 파일

분류, 세그멘테이션, 포즈 추정 및 다른 YOLO 구현체는 MVP 범위에서 제외한다.

## 업로드 계약

### 필수 파일

| 파일 | 용도 |
| --- | --- |
| `args.yaml` | 모델, 학습 파라미터 및 실행 설정 추출 |
| `results.csv` | epoch별 손실과 평가 지표 추출 |

필수 파일이 없거나 읽을 수 없으면 Parser는 진단을 시작하지 않고 누락 파일을 사용자에게 알린다.

### 선택 파일

| 파일 | 용도 |
| --- | --- |
| `data.yaml` | 클래스 이름과 데이터셋 설정 추출 |
| `confusion_matrix.png` | 클래스 간 오탐·미탐 분석 보조 |
| `confusion_matrix_normalized.png` | 클래스별 상대 오류 분석 보조 |
| `BoxPR_curve.png` 또는 `PR_curve.png` | precision-recall 균형 분석 보조 |
| `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` 및 접두사 없는 이름 | 추론 threshold 분석 보조 |
| `results.png` | 전체 학습 추세의 시각적 확인 |
| `weights/best.pt`, `weights/last.pt` | 후보 재학습 연결을 위한 체크포인트 식별 |

선택 파일이 없으면 분석은 계속하되 `parser.warnings`에 진단 제한을 기록한다.

## Parser 출력 원칙

1. Parser는 원본 파일을 변경하지 않는다.
2. 모든 경로는 ZIP 루트를 기준으로 한 상대 경로로 기록한다.
3. 손실과 점수 필드는 YOLO 원본 열 이름과 관계없이 표준 필드명으로 변환한다.
4. 사용할 수 없는 선택 값은 임의로 추정하지 않고 `null`로 기록한다.
5. score 계열 값은 `0`에서 `1` 사이로 정규화한다.
6. epoch 번호는 원본 `results.csv` 값을 그대로 유지해 원본 로그와 대조할 수 있게 한다.
7. 그래프는 JSON에 삽입하지 않고 `artifacts`에 경로와 사용 가능 여부만 기록한다.
8. 체크포인트는 JSON에 삽입하거나 진단 과정에서 실행하지 않는다.

## 데이터 구조

`evaluation.json`은 다음 최상위 영역으로 구성된다.

| 영역 | 내용 |
| --- | --- |
| `source` | 입력 형식, YOLO 버전, 발견한 원본 파일 |
| `model` | 작업 유형, 모델 구조, 체크포인트 경로 |
| `training` | epoch, batch, imgsz, optimizer 및 augmentation |
| `dataset` | 클래스와 선택적 데이터 통계 |
| `metrics` | 최고 epoch, 전체·클래스별·epoch별 지표 |
| `artifacts` | 그래프와 체크포인트의 상대 경로 |
| `parser` | Parser 버전과 누락·변환 경고 |

정확한 필드와 타입은 [`../schemas/evaluation.schema.json`](../schemas/evaluation.schema.json), 예시는 [`../examples/evaluation.example.json`](../examples/evaluation.example.json)을 따른다.

`dataset.config_path`는 ZIP 안에 실제 `data.yaml`이 있을 때의 상대 경로다. `dataset.config_reference`는 파일 포함 여부와 관계없이 `args.yaml`의 `data` 값을 보존한다. 로컬 절대 경로나 이름만으로 서버 파일에 접근하지 않는다.

## 진단 가능 수준

### `args.yaml`과 `results.csv`만 있는 경우

- 학습 정체 및 발산 후보
- 전체 지표 기준 과적합·과소적합 후보
- precision과 recall 불균형
- 최고 성능 epoch와 불필요한 추가 학습

### 데이터 통계와 클래스별 지표가 추가된 경우

- 클래스 불균형
- 특정 클래스의 오탐·미탐
- 작은 객체 비중과 성능 저하의 연관성
- 데이터 또는 라벨 문제 후보

데이터나 라벨을 직접 검사하지 않은 상태에서는 라벨 누락, 데이터 누수 또는 작은 객체 문제가 확정됐다고 표현하지 않는다.

## 보안 제한

- ZIP 경로 순회(`../`)와 절대 경로를 거부한다.
- 압축 해제 파일 수, 개별 파일 크기, 총 압축 해제 크기를 제한한다.
- 심볼릭 링크와 실행 파일을 거부한다.
- YAML은 안전 로더로 읽는다.
- 업로드된 `.pt` 체크포인트는 Parser 및 AI 진단 과정에서 로드하거나 실행하지 않는다.
- 재학습은 향후 별도의 격리 환경과 사용자 승인 절차를 통해 수행한다.

## MVP 승인 기준

- 정상적인 YOLO 결과 ZIP이 스키마에 맞는 JSON으로 변환된다.
- 필수 파일 누락 시 명확한 오류가 반환된다.
- 선택 파일 누락 시 변환은 성공하고 경고가 남는다.
- 알 수 없는 CSV 열은 무시하거나 경고하며 잘못된 값으로 매핑하지 않는다.
- 생성된 JSON이 `evaluation.schema.json` 검증을 통과한다.
