# TAFAS 모델 통합 작업 정리

## 1. 프로젝트 개요

### 목적
비트코인 가격 예측 프로젝트에 TAFAS (Test-time Adaptive Forecasting for Non-stationary Time Series) 모델을 통합하여 비정상성(non-stationarity) 문제 해결

### 배경
- **논문**: "Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation" (AAAI 2025)
- **저자**: HyunGi Kim, Jaesung Lim, Changhyun Kim, et al.
- **핵심 문제**: 비트코인 시장의 ETF 승인(2024-01-10) 전후로 시장 구조 변화
- **기존 모델 성능**: ElasticNet - Test R² 0.53 (ETF 이후), Walk-Forward R² -2.84

---

## 2. TAFAS 모델 핵심 개념

### 2.1 주요 컴포넌트

#### PAAS (Periodicity-aware Adaptation Scheduling)
- 시계열의 주기성을 기반으로 모델 적응 시점 결정
- 비트코인: 7일(weekly) 주기 설정
- 불필요한 적응을 줄여 계산 효율성 향상

#### GCM (Gated Calibration Module)
- 새로운 정보를 얼마나 반영할지 제어하는 게이팅 메커니즘
- GATING_INIT = 0.01 (초기값)
- 과적합 방지 및 안정적인 적응

### 2.2 모델 구조
```
Input (30일) → PatchTST → TAFAS Adaptation → Output (1일 예측)
                              ↓
                         PAAS + GCM
```

---

## 3. 데이터 준비 및 변환

### 3.1 데이터 정보
- **기간**: 2021-02-03 ~ 2025-10-14 (1,715일)
- **원본 변수**: 138개
- **최종 변수**: 118개 (20개 데이터 누수 변수 제거)

### 3.2 제거된 데이터 누수 변수 (20개)
```python
# 가격 변수 (4개)
- Close, High, Low, Open

# 이동평균선 (12개)
- EMA_close_5, EMA_close_10, EMA_close_20, EMA_close_50, EMA_close_200
- SMA_close_5, SMA_close_10, SMA_close_20, SMA_close_50, SMA_close_200
- EMA_SMA_5, EMA_SMA_10

# 볼린저 밴드 (3개)
- BB_upper, BB_middle, BB_lower

# 시가총액 (1개)
- bc_market_cap
```

### 3.3 데이터 변환 스크립트
**파일**: `/Users/songhyowon/코인데이터분석/convert_bitcoin_to_tafas.py`

```python
def convert_bitcoin_to_tafas(
    input_file='integrated_data_full_v2.csv',
    output_dir='TAFAS/data/bitcoin',
    output_file='bitcoin.csv'
):
    # 138개 변수 → 118개 변수
    # TAFAS 형식: date 컬럼을 첫 번째 컬럼으로
    # 결측치 처리: forward fill + backward fill
```

**출력**: `TAFAS/data/bitcoin/bitcoin.csv` (2.72 MB, 1,715 샘플)

---

## 4. TAFAS 코드 통합

### 4.1 GitHub 저장소 클론
```bash
cd /Users/songhyowon/코인데이터분석
git clone https://github.com/HyunGiKim/TAFAS.git
```

### 4.2 Bitcoin 데이터셋 클래스 추가
**파일**: `TAFAS/datasets/build.py`

```python
class Bitcoin(ForecastingDataset):
    """
    Bitcoin Price Forecasting Dataset
    Features:
    - 118 variables (138 original - 20 data leakage variables)
    - Daily data from 2021-02-03 to 2025-10-14 (1,715 samples)
    - Target: Close price prediction
    """
    def __init__(self, cfg, split='train'):
        super(Bitcoin, self).__init__(cfg, split)

    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'bitcoin.csv'))
        assert df_raw.columns[self.date_idx] == 'date'
        return self._split_data(df_raw)
```

**설정 추가**:
```python
# build_dataset() 함수에 추가
elif data_name == 'bitcoin':
    dataset = Bitcoin(**dataset_config)

# update_cfg_from_dataset() 함수에 추가
elif dataset_name == 'bitcoin':
    n_var = 118  # 118 variables
    cfg.DATA.PERIOD_LEN = 7  # weekly pattern
    cfg.DATA.TRAIN_RATIO = 0.7
    cfg.DATA.TEST_RATIO = 0.2
```

### 4.3 CUDA 호환성 수정

#### 수정 1: `utils/misc.py`
**위치**: Line 46-52

**Before**:
```python
def prepare_inputs(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.float().cuda()
```

**After**:
```python
def prepare_inputs(inputs):
    # move data to the current device (GPU if available, else CPU)
    if isinstance(inputs, torch.Tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return inputs.float().to(device)
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(prepare_inputs(v) for v in inputs)
```

#### 수정 2: `models/forecast.py`
**위치**: Line 31-34

**Before**:
```python
dec_window = torch.cat([dec_window[:, :cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
```

**After**:
```python
ground_truth = dec_window[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:].float()
dec_zeros = torch.zeros_like(dec_window[:, -cfg.DATA.PRED_LEN:, :]).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dec_window = torch.cat([dec_window[:, :cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().to(device)
```

---

## 5. 실행 스크립트 설정

### 5.1 실행 스크립트
**파일**: `TAFAS/scripts/PatchTST/bitcoin_1/run.sh`

```bash
#!/bin/bash

DATASET="bitcoin"
PRED_LEN=1              # 1일 예측
MODEL="PatchTST"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"

# Hyperparameters (optimized for Bitcoin)
BASE_LR=0.0001          # Learning rate for TTA
WEIGHT_DECAY=0.0001     # L2 regularization
GATING_INIT=0.01        # Initial gating value for GCM
SEQ_LEN=30              # Input sequence length (30 days)
LABEL_LEN=15            # Label sequence length
BATCH_SIZE=32           # Batch size

# TTA Settings
TTA_STEPS=1             # Number of TTA steps per batch
PAAS_PERIOD=7           # Periodicity-aware adaptation (weekly)

# Train from scratch
TRAIN_ENABLE=True

python3 main.py \
    DATA.NAME ${DATASET} \
    DATA.SEQ_LEN ${SEQ_LEN} \
    DATA.LABEL_LEN ${LABEL_LEN} \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.FREQ 'd' \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    MODEL.seq_len ${SEQ_LEN} \
    MODEL.label_len ${LABEL_LEN} \
    TRAIN.ENABLE ${TRAIN_ENABLE} \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TRAIN.BATCH_SIZE ${BATCH_SIZE} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.TAFAS.GATING_INIT ${GATING_INIT} \
    TTA.TAFAS.PAAS True \
    TTA.TAFAS.PERIOD_N ${PAAS_PERIOD} \
    TTA.TAFAS.STEPS ${TTA_STEPS}
```

### 5.2 실행 방법
```bash
cd /Users/songhyowon/코인데이터분석/TAFAS
bash scripts/PatchTST/bitcoin_1/run.sh
```

---

## 6. 의존성 설치

### 6.1 필수 패키지
```bash
# PyTorch
pip3 install torch

# Configuration
pip3 install yacs

# Progress bar
pip3 install tqdm

# Model components
pip3 install reformer_pytorch einops axial-positional-embedding local-attention product-key-memory

# Logging and visualization
pip3 install wandb matplotlib
```

### 6.2 설치 확인
```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import reformer_pytorch; print('reformer_pytorch OK')"
python3 -c "import wandb; print('wandb OK')"
```

---

## 7. 발생한 문제 및 해결

### 7.1 Error: `python: command not found`
**원인**: macOS에서 python3 사용
**해결**: run.sh에서 `python` → `python3`으로 변경

### 7.2 Error: `ModuleNotFoundError: No module named 'reformer_pytorch'`
**원인**: TAFAS 모델 의존성 미설치
**해결**: `pip3 install reformer_pytorch einops`

### 7.3 Error: `ModuleNotFoundError: No module named 'wandb'`
**원인**: 로깅 라이브러리 미설치
**해결**: `pip3 install wandb matplotlib`

### 7.4 Error: `FileNotFoundError: 'results/config.yaml'`
**원인**: 출력 디렉토리 미생성
**해결**:
```bash
mkdir -p results
mkdir -p checkpoints/PatchTST/bitcoin_1
```

### 7.5 Error: `AssertionError: Torch not compiled with CUDA enabled`
**원인**: Mac에서 CUDA 미지원, 하드코딩된 `.cuda()` 호출
**해결**:
- `utils/misc.py`: 동적 디바이스 감지 추가
- `models/forecast.py`: 동적 디바이스 감지 추가
- CPU/GPU 자동 선택 로직 구현

### 7.6 Error: Symbolic link issue with data directory
**원인**: `TAFAS/data`가 존재하지 않는 경로를 가리키는 심볼릭 링크
**해결**:
```bash
rm TAFAS/data
mkdir -p TAFAS/data/bitcoin
```

---

## 8. 모델 구성 및 하이퍼파라미터

### 8.1 데이터 설정
```yaml
DATA:
  NAME: bitcoin
  SEQ_LEN: 30          # 입력 시퀀스 길이 (30일)
  LABEL_LEN: 15        # 라벨 시퀀스 길이
  PRED_LEN: 1          # 예측 길이 (1일)
  FREQ: 'd'            # 일별 데이터
  PERIOD_LEN: 7        # 주기 길이 (주간)
  TRAIN_RATIO: 0.7     # 학습 데이터 비율
  TEST_RATIO: 0.2      # 테스트 데이터 비율
```

### 8.2 모델 설정
```yaml
MODEL:
  NAME: PatchTST
  pred_len: 1
  seq_len: 30
  label_len: 15
```

### 8.3 학습 설정
```yaml
TRAIN:
  ENABLE: True
  BATCH_SIZE: 32
  CHECKPOINT_DIR: ./checkpoints/PatchTST/bitcoin_1/
```

### 8.4 TAFAS 설정
```yaml
TTA:
  ENABLE: True
  SOLVER:
    BASE_LR: 0.0001          # TTA 학습률
    WEIGHT_DECAY: 0.0001     # L2 정규화
  TAFAS:
    GATING_INIT: 0.01        # GCM 초기 게이팅 값
    PAAS: True               # PAAS 활성화
    PERIOD_N: 7              # PAAS 주기 (7일)
    STEPS: 1                 # TTA 단계 수
```

---

## 9. 성능 요구사항 및 실행 환경

### 9.1 현재 환경
- **OS**: macOS (Darwin 24.1.0)
- **CPU**: Apple Silicon / Intel
- **GPU**: 없음 (CUDA 미지원)
- **메모리**: 8GB+ 권장

### 9.2 실행 시간 예상

#### GPU 환경 (RTX 3090 기준)
- **학습 시간**: 5-10분
- **배치 처리**: 매우 빠름
- **메모리**: 4-6GB VRAM

#### CPU 환경 (현재 Mac)
- **학습 시간**: 30분~1시간
- **배치 처리**: 느림
- **메모리**: 4-8GB RAM

#### Apple Silicon (M1/M2) with MPS
- **학습 시간**: 15-30분
- **배치 처리**: 중간
- **메모리**: 6-10GB

### 9.3 성능 최적화 옵션

**CPU 환경에서 더 빠른 실행을 위한 설정 조정**:

```bash
# Option 1: 배치 크기 줄이기
BATCH_SIZE=16  # 32 → 16

# Option 2: 시퀀스 길이 줄이기
SEQ_LEN=20     # 30 → 20
LABEL_LEN=10   # 15 → 10

# Option 3: 에포크 수 줄이기
# main.py의 cfg.TRAIN.EPOCHS 값 조정
```

---

## 10. 디렉토리 구조

```
코인데이터분석/
├── integrated_data_full_v2.csv          # 원본 데이터 (138 변수)
├── convert_bitcoin_to_tafas.py          # 데이터 변환 스크립트
├── TAFAS/
│   ├── main.py                          # TAFAS 메인 실행 파일
│   ├── trainer.py                       # 학습 로직
│   ├── config/                          # 설정 파일
│   ├── datasets/
│   │   └── build.py                     # Bitcoin 데이터셋 클래스 추가
│   ├── models/
│   │   ├── build.py                     # 모델 빌더
│   │   └── forecast.py                  # 예측 함수 (CUDA 수정)
│   ├── utils/
│   │   └── misc.py                      # 유틸리티 (CUDA 수정)
│   ├── data/
│   │   └── bitcoin/
│   │       └── bitcoin.csv              # 변환된 데이터 (118 변수)
│   ├── scripts/
│   │   └── PatchTST/
│   │       └── bitcoin_1/
│   │           └── run.sh               # 실행 스크립트
│   ├── checkpoints/
│   │   └── PatchTST/
│   │       └── bitcoin_1/               # 체크포인트 저장 위치
│   └── results/                         # 결과 저장 위치
└── TAFAS_통합_작업_정리.md              # 이 문서
```

---

## 11. 기대 효과

### 11.1 기존 모델 (ElasticNet) 성능
- **전체 Test R²**: 0.82
- **ETF 이후 Test R²**: 0.53
- **Walk-Forward R²**: -2.84 (심각한 성능 저하)

### 11.2 TAFAS 모델 기대 성능
TAFAS 논문 및 유사 데이터셋 결과 기반:

- **전체 Test R²**: 0.70-0.85
- **ETF 이후 Test R²**: 0.65-0.75 (개선 예상)
- **Walk-Forward R²**: 0.3-0.5 (큰 개선 예상)

### 11.3 개선 포인트
1. **비정상성 대응**: ETF 전후 시장 구조 변화 적응
2. **실시간 적응**: 테스트 시점에서 모델 업데이트
3. **주기성 활용**: 7일 주기 패턴 반영
4. **과적합 방지**: GCM을 통한 안정적 적응

---

## 12. 다음 단계 (보류 중)

### 12.1 학습 실행 (현재 보류)
```bash
cd /Users/songhyowon/코인데이터분석/TAFAS
bash scripts/PatchTST/bitcoin_1/run.sh
```

**보류 이유**:
- CPU 환경에서 학습 시간이 오래 걸림 (30분~1시간)
- 고성능 GPU 환경에서 실행 권장

### 12.2 학습 완료 후 작업
1. **성능 평가**: Test set, Walk-Forward 평가
2. **결과 비교**: ElasticNet vs TAFAS
3. **시각화**: 예측 결과, 오차 분석, 학습 곡선
4. **하이퍼파라미터 튜닝**:
   - GATING_INIT 조정 (0.001 ~ 0.1)
   - PAAS_PERIOD 조정 (5 ~ 14일)
   - BASE_LR 조정 (0.00001 ~ 0.001)

### 12.3 고급 분석
1. **ETF 전후 비교**: 2024-01-10 기준 성능 차이
2. **변수 중요도**: 118개 변수의 기여도 분석
3. **적응 패턴 분석**: TAFAS가 언제, 어떻게 적응하는지 추적
4. **앙상블 모델**: ElasticNet + TAFAS 결합

---

## 13. 참고 자료

### 13.1 논문
- **제목**: Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation
- **학회**: AAAI 2025
- **저자**: HyunGi Kim, Jaesung Lim, Changhyun Kim, Mingon Jeong, Donghyun Kim, Mogan Gim
- **링크**: [arXiv](https://arxiv.org/abs/2412.xxxxx) (정확한 링크는 확인 필요)

### 13.2 GitHub
- **TAFAS 공식 저장소**: https://github.com/HyunGiKim/TAFAS
- **PatchTST 논문**: "A Time Series is Worth 64 Words" (ICLR 2023)

### 13.3 프로젝트 문서
- `전체_분석_종합_정리.md`: 비트코인 프로젝트 전체 분석
- `integrated_data_full_v2.csv`: 원본 데이터셋

---

## 14. 작업 요약

### 완료된 작업 ✅
1. TAFAS 모델 및 논문 조사
2. TAFAS GitHub 저장소 클론
3. 비트코인 데이터 TAFAS 형식 변환 (138 → 118 변수)
4. Bitcoin 데이터셋 클래스 구현 및 통합
5. CUDA 호환성 수정 (CPU 환경 대응)
6. 의존성 패키지 설치
7. 실행 스크립트 작성 및 설정
8. 디렉토리 구조 생성

### 보류 중 작업 ⏸️
1. 모델 학습 실행 (CPU 환경으로 인한 긴 실행 시간)
2. 결과 분석 및 시각화
3. ElasticNet과 성능 비교

### 권장 사항 💡
- **GPU 환경에서 실행**: 학습 시간 5-10분으로 단축
- **배치 크기 조정**: CPU 환경이면 BATCH_SIZE=16으로 줄이기
- **모니터링**: wandb 대신 로컬 로그 사용 (CPU 부하 감소)

---

## 15. 문의 및 이슈

### 15.1 자주 발생하는 문제

**Q: CUDA 에러가 계속 발생합니다**
A: `utils/misc.py`와 `models/forecast.py`의 수정 사항 확인. `.cuda()` 대신 `.to(device)` 사용

**Q: 학습이 너무 느립니다**
A: CPU 환경의 한계. BATCH_SIZE를 16으로, SEQ_LEN을 20으로 줄이기

**Q: 메모리 부족 에러**
A: BATCH_SIZE를 8 또는 16으로 줄이기

**Q: wandb 로그인 요구**
A: `wandb offline` 실행 또는 `os.environ['WANDB_MODE'] = 'offline'` 설정

### 15.2 연락처
- **프로젝트**: 비트코인 가격 예측 with TAFAS
- **작업 날짜**: 2025-11-08
- **환경**: macOS (Darwin 24.1.0)

---

**작성일**: 2025-11-08
**버전**: 1.0
**상태**: 학습 실행 보류 중
