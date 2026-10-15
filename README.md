# AI Model Doctor

Ultralytics YOLO 객체 탐지 학습 결과를 표준 JSON으로 변환하고 진단하기 위한 프로젝트입니다.

## 현재 구현

- YOLO 결과 디렉터리 또는 ZIP 입력
- `args.yaml`과 `results.csv` 필수 파일 검증
- 안전한 ZIP 압축 해제
- 학습 설정과 epoch별 지표 정규화
- 선택 그래프와 체크포인트 목록화
- `evaluation.json` 생성
- 규칙 기반 `diagnosis.json` 생성
- OpenAI Responses API를 통한 구조화된 챗봇 응답 생성

## Parser 실행

```bash
python3 -m model_doctor.parser runs/detect/train -o evaluation.json
```

ZIP 파일도 같은 방식으로 사용할 수 있습니다.

```bash
python3 -m model_doctor.parser yolo-result.zip -o evaluation.json
```

## 1차 진단 실행

현재는 학습 기록 충분성, 전체 Precision·Recall 불균형, 학습 정체, 과적합 의심 및 최적 epoch 이후 성능 하락 규칙을 지원합니다.

```bash
python3 -m model_doctor.diagnosis evaluation.json -o diagnosis.json
```

## AI 진단 설명 생성

OpenAI API 키를 환경변수로 설정한 뒤 평가와 진단 결과를 함께 전달합니다. 기본 모델은 `gpt-5.4-mini`이며 `OPENAI_MODEL`로 변경할 수 있습니다.

```bash
export OPENAI_API_KEY="your-api-key"
python3 -m model_doctor.llm evaluation.json diagnosis.json "전체 결과를 요약해줘" -o chat-response.json
```

응답은 `schemas/chat-response.schema.json`을 따르며, 챗봇은 규칙 엔진의 판정을 설명할 뿐 상태를 다시 판정하거나 모델 설정을 변경하지 않습니다.

## 테스트

```bash
python3 -m unittest discover -v
```

입력 규격은 [`docs/input-contract.md`](docs/input-contract.md)를 참고하세요.
