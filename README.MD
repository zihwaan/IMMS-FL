# Federated Anomaly Detection with WADI Dataset

## 📋 실험 개요

본 레포는 **계층적 연합학습이 WADI처럼 불균형·분포가 요동치는 시계열에서도 실제로 도움이 되는가**라는 핵심 질문에 답하기 위한 실험입니다.

### 🎯 실험 목표

1. **집계 알고리즘 효과 검증**: FedAvg, FedProx, FedAdam의 성능 비교
2. **통신 효율성 검증**: Frozen Encoder 전략의 통신량 절약 효과
3. **분포 변동 대응력 검증**: Wasserstein Distance 기반 분포 변화 감지 및 대응

### 🏗️ 실험 설계

#### 데이터 분할 전략
```
WADI 데이터 → 3개 Edge 분리:
- Edge 0: 유량/압력 공격 집중 + 정상 데이터 1/3
- Edge 1: 센서/은닉 공격 집중 + 정상 데이터 1/3  
- Edge 2: 품질/수압 공격 집중 + 정상 데이터 1/3
```

#### 모델 아키텍처
- **Anomaly Transformer**: 시계열 이상 탐지에 특화된 Transformer 기반 모델
- **입력 차원**: 123개 WADI 센서 피처
- **시퀀스 길이**: 60 시점
- **출력**: 정상(0) vs 이상(1) 이진 분류

#### 실험 조합 (총 18가지)
```
집계 알고리즘 (3개) × Encoder 전략 (2개) × Edge 수 (3개 고정)
- FedAvg, FedProx, FedAdam
- Frozen Encoder vs Full Fine-tuning
- 3개 Edge 고정
```

#### 분포 변동 시나리오
- **라운드 50**에서 Edge 2에 급격한 분포 변화 적용
- **스케일 2배 증가**로 공격 패턴 변화 시뮬레이션
- **Wasserstein Distance > 0.2**시 재배치 트리거

## 🚀 빠른 시작

### 환경 설정

```bash
# 1. 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터 확인
# WADI_FINAL_DATASET.csv 파일이 프로젝트 루트에 있는지 확인
```

### 단일 실험 실행

```bash
# FedAvg + Full Model로 100 라운드 실험
python experiment_runner.py --algorithm fedavg --encoder full --rounds 100

# FedProx + Frozen Encoder로 50 라운드 실험  
python experiment_runner.py --algorithm fedprox --encoder frozen --rounds 50

# FedAdam + Full Model로 실험
python experiment_runner.py --algorithm fedadam --encoder full --rounds 100
```

### 전체 실험 실행 (권장)

```bash
# 모든 조합 실험 (총 6개: 3 알고리즘 × 2 Encoder 전략)
python experiment_runner.py --algorithm all --encoder both --rounds 100

# 특정 알고리즘 모든 Encoder 전략
python experiment_runner.py --algorithm fedprox --encoder both --rounds 100
```

### 결과 분석만 실행

```bash
# 기존 실험 결과 분석 및 시각화
python experiment_runner.py --analyze_only
```

## 📊 실험 결과 이해

### 출력 파일 구조

```
results/
├── fedavg_full_YYYYMMDD_HHMMSS_history.csv     # 실험별 상세 로그
├── fedprox_frozen_YYYYMMDD_HHMMSS_history.csv  # 라운드별 메트릭
├── all_experiments_YYYYMMDD_HHMMSS.json        # 전체 실험 결과
├── experiment_results.png                      # 시각화 차트
└── experiment_report.md                        # 요약 보고서
```

### 주요 메트릭

1. **성능 메트릭**
   - `val_f1_avg`: 검증 F1 점수 (주요 지표)
   - `val_accuracy_avg`: 검증 정확도
   - `val_loss_avg`: 검증 손실

2. **통신 메트릭**
   - Frozen Encoder: ~50% 통신량 감소 예상
   - 파라미터 수: Detection Head만 전송

3. **분포 변동 메트릭**
   - `wasserstein_distance`: 분포 변화 정도
   - `shift_detected`: 임계값(0.2) 초과 여부
   - `rounds_to_recover`: F1 점수 회복까지 라운드 수

## 🔍 세부 구현 설명

### 핵심 컴포넌트

#### 1. 데이터 로더 (`data_loader.py`)
```python
# WADI 데이터를 연합학습용으로 분할
loader = WADIDataLoader("WADI_FINAL_DATASET.csv")
X, y = loader.load_and_preprocess()
data_splits = loader.create_federated_split(X, y)
```

#### 2. Anomaly Transformer (`anomaly_transformer.py`)
```python
# Transformer 기반 이상 탐지 모델
model = create_model(input_dim=123, frozen_encoder=True)
model.freeze_encoder()  # Frozen Encoder 전략
```

#### 3. 집계 알고리즘 (`fl_algorithms.py`)
```python
# 세 가지 집계 알고리즘 구현
aggregator = create_aggregator('fedprox', device, mu=0.01)
global_weights = aggregator.aggregate(client_weights, client_samples)
```

#### 4. Flower 통합 (`fl_client.py`, `fl_server.py`)
```python
# 클라이언트와 서버는 Flower 프레임워크 기반
client = create_client(client_id, model, trainloader, valloader, device, config)
server = create_strategy(model, algorithm, config)
```

### 실험 흐름

1. **사전학습** (Frozen Encoder 사용시)
   ```python
   # 정상 데이터로 Encoder 사전학습
   pretrain_encoder(model, pretrain_dataloader, device, epochs=10)
   ```

2. **연합학습 라운드**
   ```python
   for round in range(100):
       # 클라이언트 훈련
       client_weights = train_clients(global_weights)
       
       # 집계
       global_weights = aggregator.aggregate(client_weights, samples)
       
       # 분포 변동 감지 (라운드 50)
       if round == 50:
           apply_distribution_shift(client_2)
   ```

3. **결과 수집**
   ```python
   # CSV로 라운드별 메트릭 저장  
   metrics = evaluate_global_model(global_weights)
   save_to_csv(round, metrics)
   ```

## 📈 예상 결과

### 성능 비교
- **FedAdam**: 가장 높은 F1 점수 예상 (적응적 학습률)
- **FedProx**: 불균형 데이터에서 안정적 수렴
- **FedAvg**: 기준선 성능

### 통신 효율성
- **Frozen Encoder**: ~50% 통신량 감소, 성능 하락 <1%
- **Full Model**: 기준 통신량, 최대 성능

### 분포 변동 대응
- **Wasserstein Distance > 0.2**: 자동 재배치 트리거
- **F1 회복 속도**: 재배치 적용시 2-3배 빠른 회복

## 🔧 커스텀 사용법

### 커스텀 모델 설정

```python
# anomaly_transformer.py 수정
model_config = {
    'd_model': 512,        # 모델 차원 증가
    'n_heads': 16,         # 어텐션 헤드 수 증가  
    'n_layers': 8,         # 레이어 수 증가
    'd_ff': 2048,          # Feed-forward 차원
    'dropout': 0.2         # 드롭아웃 비율
}
```

### 커스텀 집계 알고리즘 파라미터

```python
# FedProx proximal parameter 조정
python experiment_runner.py --algorithm fedprox --mu 0.05

# FedAdam 학습률 조정  
python experiment_runner.py --algorithm fedadam --eta 0.001
```

### 분포 변동 시나리오 커스터마이징

```python
# fl_server.py의 configure_fit 메소드 수정
if server_round == 30:  # 라운드 30에서 변화
    config['shift_intensity'] = 3.0  # 강도 3배
```

## 🐛 문제 해결


1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 감소
   # data_loader.py에서 batch_size=16으로 수정
   ```

2. **Flower 연결 오류**
   ```bash
   # 포트 충돌시 포트 변경
   python experiment_runner.py --server_address 0.0.0.0:8081
   ```

3. **데이터 로딩 오류**
   ```bash
   # WADI_FINAL_DATASET.csv 경로 확인
   python experiment_runner.py --data_path /path/to/WADI_FINAL_DATASET.csv
   ```


## 📚 참고 문헌

1. **Anomaly Transformer**: "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"
2. **FedAvg**: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
3. **FedProx**: "Federated Optimization in Heterogeneous Networks"
4. **FedAdam**: "Adaptive Federated Optimization"
5. **WADI Dataset**: "A Multi-stage Cyber Attack Dataset"


---

## ⚡ 실험 실행 체크리스트

- [ ] Python 3.8+ 설치 확인
- [ ] 가상환경 생성 및 활성화
- [ ] `pip install -r requirements.txt` 실행
- [ ] `WADI_FINAL_DATASET.csv` 파일 확인
- [ ] `python experiment_runner.py --algorithm all --encoder both --rounds 100` 실행
- [ ] `results/` 폴더에서 결과 확인
- [ ] `experiment_report.md`에서 요약 결과 확인
