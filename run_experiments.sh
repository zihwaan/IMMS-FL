#!/bin/bash

# IMMS-FL 실험 실행 스크립트
# WADI 데이터셋에서 계층적 연합학습 실험을 수행합니다.

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 설정 변수
DATASET_PATH="WADI_FINAL_DATASET.csv"
WINDOW_SIZE=100
BATCH_SIZE=32
NUM_ROUNDS=100
EXPERIMENT_LOG_DIR="experiment_logs"

# 실험 설정
MODELS=("duogat" "anomaly_transformer")
STRATEGIES=("FedAvg" "FedProx" "FedAdam")
FREEZE_OPTIONS=("false" "true")

# 도움말 함수
show_help() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -h, --help              이 도움말을 출력"
    echo "  -d, --dataset PATH      WADI 데이터셋 경로 (기본값: WADI_FINAL_DATASET.csv)"
    echo "  -r, --rounds NUM        연합학습 라운드 수 (기본값: 100)"
    echo "  -b, --batch-size NUM    배치 크기 (기본값: 32)"
    echo "  -w, --window-size NUM   윈도우 크기 (기본값: 100)"
    echo "  --check-only            의존성 검사만 수행"
    echo "  --validate-only         데이터 검증만 수행"
    echo "  --summary-only          결과 요약만 생성"
    echo "  --quick                 빠른 실험 (각 조합당 10라운드)"
    echo "  --single MODEL STRATEGY 단일 실험 실행"
    echo ""
    echo "예시:"
    echo "  $0                                    # 전체 실험 실행"
    echo "  $0 --quick                           # 빠른 실험"
    echo "  $0 --single duogat FedAvg            # DuoGAT + FedAvg만 실행"
    echo "  $0 --check-only                      # 의존성만 확인"
}

# 명령행 인수 파싱
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dataset)
                DATASET_PATH="$2"
                shift 2
                ;;
            -r|--rounds)
                NUM_ROUNDS="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -w|--window-size)
                WINDOW_SIZE="$2"
                shift 2
                ;;
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --summary-only)
                SUMMARY_ONLY=true
                shift
                ;;
            --quick)
                NUM_ROUNDS=10
                log_info "빠른 실험 모드: 각 실험을 10라운드로 제한"
                shift
                ;;
            --single)
                SINGLE_MODEL="$2"
                SINGLE_STRATEGY="$3"
                shift 3
                ;;
            *)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 전제 조건 확인
check_prerequisites() {
    log_info "전제 조건 확인 중..."
    
    # Python 설치 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되어 있지 않습니다."
        exit 1
    fi
    
    # 필수 패키지 확인
    if ! python3 experiment_utils.py check_deps; then
        log_error "필수 Python 패키지가 누락되었습니다."
        log_info "다음 명령으로 설치하세요: pip install -r requirements.txt"
        exit 1
    fi
    
    log_success "모든 전제 조건이 충족되었습니다."
}

# 데이터 검증
validate_dataset() {
    log_info "데이터셋 검증 중: $DATASET_PATH"
    
    if [[ ! -f "$DATASET_PATH" ]]; then
        log_error "데이터셋 파일을 찾을 수 없습니다: $DATASET_PATH"
        exit 1
    fi
    
    if ! python3 experiment_utils.py validate_data "$DATASET_PATH"; then
        log_error "데이터셋 검증 실패"
        exit 1
    fi
    
    log_success "데이터셋 검증 완료"
}

# 실험 디렉토리 준비
prepare_experiment_directory() {
    log_info "실험 디렉토리 준비 중..."
    
    mkdir -p "$EXPERIMENT_LOG_DIR"
    
    # 기존 로그 백업 (옵션)
    if [[ -n "$(ls -A $EXPERIMENT_LOG_DIR 2>/dev/null)" ]]; then
        BACKUP_DIR="${EXPERIMENT_LOG_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        log_warning "기존 실험 로그를 백업합니다: $BACKUP_DIR"
        mv "$EXPERIMENT_LOG_DIR" "$BACKUP_DIR"
        mkdir -p "$EXPERIMENT_LOG_DIR"
    fi
    
    log_success "실험 디렉토리 준비 완료"
}

# 단일 실험 실행
run_single_experiment() {
    local model=$1
    local strategy=$2
    local freeze_encoder=$3
    
    local exp_name="${model}_${strategy}"
    if [[ "$freeze_encoder" == "true" ]]; then
        exp_name="${exp_name}_frozen"
    fi
    
    log_info "실험 시작: $exp_name"
    
    local cmd="python3 fl_run.py --model $model --strategy $strategy --num_rounds $NUM_ROUNDS --csv_path $DATASET_PATH"
    
    if [[ "$freeze_encoder" == "true" ]]; then
        cmd="$cmd --freeze_encoder"
    fi
    
    # 실험 실행
    if $cmd; then
        log_success "실험 완료: $exp_name"
        return 0
    else
        log_error "실험 실패: $exp_name"
        return 1
    fi
}

# 전체 실험 실행
run_all_experiments() {
    log_info "전체 실험 실행 시작"
    log_info "총 실험 수: $((${#MODELS[@]} * ${#STRATEGIES[@]} * ${#FREEZE_OPTIONS[@]}))"
    
    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    
    # 실험 실행
    for model in "${MODELS[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            for freeze in "${FREEZE_OPTIONS[@]}"; do
                total_experiments=$((total_experiments + 1))
                
                echo ""
                log_info "실험 진행률: $total_experiments / $((${#MODELS[@]} * ${#STRATEGIES[@]} * ${#FREEZE_OPTIONS[@]}))"
                
                if run_single_experiment "$model" "$strategy" "$freeze"; then
                    successful_experiments=$((successful_experiments + 1))
                else
                    failed_experiments=$((failed_experiments + 1))
                    log_warning "실험 실패하였지만 계속 진행합니다..."
                fi
                
                # 실험 간 잠시 대기
                sleep 2
            done
        done
    done
    
    echo ""
    log_info "=== 실험 실행 완료 ==="
    log_info "총 실험 수: $total_experiments"
    log_success "성공한 실험: $successful_experiments"
    if [[ $failed_experiments -gt 0 ]]; then
        log_error "실패한 실험: $failed_experiments"
    fi
}

# 단일 실험 실행 (--single 옵션)
run_specified_experiment() {
    log_info "지정된 실험 실행: $SINGLE_MODEL + $SINGLE_STRATEGY"
    
    # 유효성 검사
    local valid_model=false
    for model in "${MODELS[@]}"; do
        if [[ "$model" == "$SINGLE_MODEL" ]]; then
            valid_model=true
            break
        fi
    done
    
    if [[ "$valid_model" == false ]]; then
        log_error "유효하지 않은 모델: $SINGLE_MODEL"
        log_info "사용 가능한 모델: ${MODELS[*]}"
        exit 1
    fi
    
    local valid_strategy=false
    for strategy in "${STRATEGIES[@]}"; do
        if [[ "$strategy" == "$SINGLE_STRATEGY" ]]; then
            valid_strategy=true
            break
        fi
    done
    
    if [[ "$valid_strategy" == false ]]; then
        log_error "유효하지 않은 전략: $SINGLE_STRATEGY"
        log_info "사용 가능한 전략: ${STRATEGIES[*]}"
        exit 1
    fi
    
    # Frozen/Unfrozen 둘 다 실행
    log_info "Unfrozen Encoder 실험 실행..."
    run_single_experiment "$SINGLE_MODEL" "$SINGLE_STRATEGY" "false"
    
    log_info "Frozen Encoder 실험 실행..."
    run_single_experiment "$SINGLE_MODEL" "$SINGLE_STRATEGY" "true"
    
    log_success "지정된 실험 완료"
}

# 실험 결과 요약
generate_summary() {
    log_info "실험 결과 요약 생성 중..."
    
    if python3 experiment_utils.py generate_summary; then
        log_success "실험 결과 요약 생성 완료"
    else
        log_warning "실험 결과 요약 생성 실패"
    fi
}

# 메인 함수
main() {
    echo ""
    log_info "IMMS-FL 연합학습 실험 시작"
    log_info "=========================================="
    
    # 명령행 인수 파싱
    parse_arguments "$@"
    
    # 설정 출력
    log_info "실험 설정:"
    log_info "  데이터셋: $DATASET_PATH"
    log_info "  라운드 수: $NUM_ROUNDS"
    log_info "  배치 크기: $BATCH_SIZE"
    log_info "  윈도우 크기: $WINDOW_SIZE"
    
    # 조건부 실행
    if [[ "$CHECK_ONLY" == true ]]; then
        check_prerequisites
        log_success "의존성 확인 완료"
        exit 0
    fi
    
    if [[ "$VALIDATE_ONLY" == true ]]; then
        validate_dataset
        log_success "데이터 검증 완료"
        exit 0
    fi
    
    if [[ "$SUMMARY_ONLY" == true ]]; then
        generate_summary
        exit 0
    fi
    
    # 일반적인 실험 실행 흐름
    check_prerequisites
    validate_dataset
    prepare_experiment_directory
    
    # 실험 실행
    if [[ -n "$SINGLE_MODEL" && -n "$SINGLE_STRATEGY" ]]; then
        run_specified_experiment
    else
        run_all_experiments
    fi
    
    # 결과 요약
    generate_summary
    
    echo ""
    log_success "모든 실험이 완료되었습니다!"
    log_info "실험 로그는 $EXPERIMENT_LOG_DIR 디렉토리에서 확인할 수 있습니다."
}

# 스크립트 실행
main "$@"