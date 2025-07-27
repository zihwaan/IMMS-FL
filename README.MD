# IMMS-FL: 산업용 다중센서 다중모델 연합학습

## 📖 프로젝트 개요

IMMS-FL(Industrial Multi-sensor Multi-model federated learning)은 **산업용 시계열 이상 탐지**를 위한 **계층적 연합학습 시스템**입니다. 

본 프로젝트는 **WADI(Water Distribution) 데이터셋**을 사용하여 다음과 같은 핵심 질문에 답합니다:
> "**계층적 연합학습이 WADI처럼 불균형·분포가 요동치는 시계열에서도 실제로 도움이 되는가?**"

### 🎯 주요 특징

- **다중 모델 지원**: DuoGAT(그래프 신경망)과 Anomaly Transformer(시계열 트랜스포머) 비교
- **연합학습 전략**: FedAvg, FedProx, FedAdam 세 가지 집계 알고리즘
- **Frozen Encoder**: 통신 효율성을 위한 인코더 고정 전략
- **분포 변화 대응**: Wasserstein Distance를 통한 실시간 분포 변화 감지
- **Edge별 특화**: 공격 유형별로 3개 Edge에 분산된 불균형 데이터 처리

## 🏗️ 시스템 구조

```
📂 IMMS-FL/
├── 📄 fl_run.py              # 메인 연합학습 실행 스크립트
├── 📄 data_loader.py         # WADI 데이터 로더 및 전처리
├── 📄 experiment_utils.py    # 모델 어댑터 및 실험 유틸리티
├── 📄 run_experiments.sh     # 배치 실험 실행 스크립트
├── 📄 requirements.txt       # Python 의존성 패키지
├── 📂 DuoGAT/               # DuoGAT 모델 
├── 📂 Anomaly-Transformer/   # Anomaly Transformer 모델
└── 📄 WADI_FINAL_DATASET.csv # WADI 데이터셋 (106MB)
```

### 데이터 분할 전략

각 Edge는 서로 다른 센서 그룹을 담당합니다:

- **Edge 0 (유량/압력)**: FIT, DPIT, PIT, FIC 센서군
- **Edge 1 (센서/은닉)**: AIT 센서군 및 은닉 공격 패턴
- **Edge 2 (품질/수압)**: 수질 관련 센서 및 수압 제어 시스템

## 🚀 빠른 시작

### 1. 환경 설정

```bash
cd IMMS-FL

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

WADI 데이터셋이 프로젝트 루트에 `WADI_FINAL_DATASET.csv`로 있는지 확인하세요.

### 3. 단일 실험 실행

```bash
# DuoGAT + FedAvg로 기본 실험 실행
python fl_run.py --model duogat --strategy FedAvg --num_rounds 100

# Frozen Encoder 전략으로 실행
python fl_run.py --model anomaly_transformer --strategy FedProx --freeze_encoder --num_rounds 50

# 사용자 정의 데이터셋 경로
python fl_run.py --model duogat --strategy FedAdam --csv_path /path/to/your/dataset.csv
```

### 4. 배치 실험 실행

```bash
# 모든 모델과 전략 조합으로 실험 실행
chmod +x run_experiments.sh
./run_experiments.sh

# 특정 조건만 실험
./run_experiments.sh --model duogat --strategy FedAvg

# 의존성만 확인
./run_experiments.sh --check-only

# 데이터 검증만 실행
./run_experiments.sh --validate-only
```

## 📊 실험 설정

### 모델 비교

| 모델 | 특징 | 강점 |
|------|------|------|
| **DuoGAT** | 이중 그래프 어텐션 네트워크 | 센서 간 시공간 인과관계 포착 |
| **Anomaly Transformer** | 시계열 트랜스포머 | 장기 의존성 학습 |

### 연합학습 전략

| 전략 | 설명 | 적용 상황 |
|------|------|----------|
| **FedAvg** | 기본 평균 집계 | 기준선 성능 측정 |
| **FedProx** | 편차 보정 집계 | 데이터 분포 차이가 클 때 |
| **FedAdam** | 적응적 학습률 | 서버 측 최적화 필요 시 |

### 실험 파라미터

```python
# 기본 설정
WINDOW_SIZE = 100       # 시계열 윈도우 크기
BATCH_SIZE = 32         # 배치 크기
NUM_ROUNDS = 100        # 연합학습 라운드 수
NUM_CLIENTS = 3         # Edge 클라이언트 수
FREEZE_ENCODER = False  # 인코더 고정 여부
```

## 📈 결과 분석

### 실험 로그

실험 결과는 `experiment_logs/` 디렉토리에 저장됩니다:

```
📂 experiment_logs/
├── 📄 duogat_FedAvg_frozen_False_YYYYMMDD_HHMMSS.csv
├── 📄 anomaly_transformer_FedProx_frozen_True_YYYYMMDD_HHMMSS.csv
└── 📄 experiment_summary_YYYYMMDD.txt
```

### 주요 메트릭

- **F1 Score**: 이상 탐지 성능의 핵심 지표
- **Accuracy**: 전체 분류 정확도
- **Loss**: 학습 손실 함수 값
- **Wasserstein Distance**: 분포 변화 감지
- **Communication Overhead**: 통신 오버헤드 (Frozen vs Full)