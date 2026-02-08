import time
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import onnxruntime as ort
import re
from scipy.io.wavfile import write

# VITS 프로젝트 모듈 임포트
import utils
import commons
from text import text_to_sequence
from text.k2j import korean2katakana
from text.j2k import japanese2korean

# ==========================================
# [설정] 테스트 환경 설정
# ==========================================
# 모델 및 설정 파일 경로
MODEL_PATH = "ko_16k.onnx"
CONFIG_PATH = "./configs/ko_16k.json"

# 테스트할 텍스트 (길이가 좀 있어야 변별력이 있음)
TEST_TEXT = """
    [KO]결론부터 말씀드리면, 이 방법은 비추천합니다. 
    에러 메시지는 사라지게 할 수 있지만, 만들어진 모델이 "고정된 길이"의 오디오만 처리하게 될 위험이 매우 크기 때문입니다.[KO]
    """

# CPU 물리 코어 수만큼 워커를 띄우는 것이 처리량(Throughput)에 가장 좋습니다.
# 시스템 안정성을 위해 1~2개 정도 남겨두는 것을 권장합니다.
#NUM_WORKERS = max(1, os.cpu_count() - 2) 
NUM_WORKERS = 16
# 테스트 요청 횟수 (충분히 많아야 평균이 정확함)
SIMULATED_CALLS = 32

# ==========================================
# 전역 변수 (각 프로세스 내부에서만 초기화됨)
# ==========================================
ort_session = None
hps = None

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = np.array(text_norm, dtype=np.int64)
    return text_norm

def init_worker():
    """
    각 프로세스(Worker)가 시작될 때 한 번 실행됩니다.
    ONNX 세션을 프로세스별로 독립적으로 로드합니다.
    """
    global ort_session, hps
    
    try:
        # [핵심] 멀티프로세싱 환경에서 스레드 경합(Context Switching)을 줄이기 위한 설정
        # 각 워커가 싱글 스레드로 집중해서 연산하는 것이 전체 처리량에는 더 유리합니다.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # 1. 하이퍼파라미터 로드
        hps = utils.get_hparams_from_file(CONFIG_PATH)

        # 2. ONNX 세션 옵션 설정 (CPU 최적화)
        sess_options = ort.SessionOptions()
        # 병렬 처리는 프로세스 레벨에서 하므로, 연산 자체는 단일 스레드로 제한
        sess_options.intra_op_num_threads = 1 
        sess_options.inter_op_num_threads = 1
        
        # 3. 모델 로드
        # print(f"   [PID-{os.getpid()}] ONNX 모델 로드 중...")
        ort_session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
        
    except Exception as e:
        print(f"❌ [PID-{os.getpid()}] 초기화 실패: {e}")

def run_warmup():
    """모델을 한 번 실행하여 메모리에 올리고 캐싱합니다."""
    global ort_session, hps
    try:
        if ort_session is None: return False
        
        # 짧은 텍스트로 웜업
        text = "아"
        stn_tst = get_text(text, hps)
        x_tst = np.expand_dims(stn_tst, axis=0)
        x_tst_lengths = np.array([x_tst.shape[1]], dtype=np.int64)
        
        inputs = {
            'text': x_tst,
            'text_lengths': x_tst_lengths,
            'noise_scale': np.array([0.667], dtype=np.float32),
            'length_scale': np.array([1.0], dtype=np.float32),
        }
        ort_session.run(None, inputs)
        return True
    except Exception as e:
        print(f"Warmup fail: {e}")
        return False

def run_inference(req_id, text):
    """실제 추론 및 성능 측정"""
    global ort_session, hps
    result = {}
    filename = f"stress_result_{req_id}.wav"
    
    try:
        # 1. 전처리
        t0 = time.time()
        
        # 정규식 처리 ( inference_onnx.py 로직 )
        text_proc = re.sub('[\n]', '', text).strip()
        if hasattr(hps.data, 'is_japanese_dataset') and hps.data.is_japanese_dataset:
            text_proc = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean2katakana(x.group(1)), text_proc)
        elif hasattr(hps.data, 'is_korean_dataset') and hps.data.is_korean_dataset:
            text_proc = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese2korean(x.group(1)), text_proc)

        stn_tst = get_text(text_proc, hps)
        x_tst = np.expand_dims(stn_tst, axis=0)
        x_tst_lengths = np.array([x_tst.shape[1]], dtype=np.int64)

        inputs = {
            'text': x_tst,
            'text_lengths': x_tst_lengths,
            'noise_scale': np.array([0.667], dtype=np.float32),
            'length_scale': np.array([1.0], dtype=np.float32),
            # 'noise_scale_w': ... (에러나면 제외)
        }

        # 2. 추론 (ONNX Runtime)
        t1 = time.time()
        audio = ort_session.run(None, inputs)[0]
        t2 = time.time()
        
        inference_pure_time = t2 - t1 # 순수 모델 연산 시간
        
        # 3. 후처리 (정규화 및 저장)
        audio = audio.squeeze()
        max_val = np.abs(audio).max()
        if max_val > 0.99:
            audio = audio / max_val * 0.95
            
        # 오디오 길이 계산
        audio_duration = len(audio) / hps.data.sampling_rate
        
        # 파일 저장 (I/O 부하는 제외하고 싶으면 주석 처리 가능)
        write(filename, hps.data.sampling_rate, audio)

        total_time = time.time() - t0 # 전처리+추론+후처리 포함

        result['status'] = 'success'
        result['rtf'] = inference_pure_time / audio_duration # RTF는 보통 순수 추론 시간 기준
        result['total_time'] = total_time
        result['audio_len'] = audio_duration
        result['pid'] = os.getpid()
        
    except Exception as e:
        result['status'] = 'fail'
        result['error'] = str(e)
        result['rtf'] = 0.0

    return result

if __name__ == "__main__":
    # 리눅스/윈도우 호환성을 위해 spawn 방식 권장
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 기존 결과 파일 정리
    os.system("rm -f stress_result_*.wav")
    
    print(f"\n>>> [System] ONNX CPU 벤치마크 시작 (Workers: {NUM_WORKERS})")
    print(f">>> [Model] {MODEL_PATH}")
    print(f">>> [Tip] 'htop'으로 CPU 사용률을 확인하세요. (Target: RTF < 0.1)")
    
    executor = ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker)

    # 1. Warm-up
    print(">>> [Warm-up] 워커 프로세스 초기화 중...")
    warmup_futures = [executor.submit(run_warmup) for _ in range(NUM_WORKERS)]
    for f in warmup_futures: 
        f.result()
    print(">>> 🚀 [Start] 부하 테스트 시작!")
    
    # 2. 측정 시작
    start_total = time.time()
    futures = [executor.submit(run_inference, i, TEST_TEXT) for i in range(SIMULATED_CALLS)]
    results = [f.result() for f in futures]
    end_total = time.time()
    
    executor.shutdown()

    # 3. 결과 분석
    print("\n>>> ================= ONNX CPU 최종 결과 ================= <<<")
    success_res = [r for r in results if r['status'] == 'success']
    fail_res = [r for r in results if r['status'] == 'fail']
    
    elapsed_wall_clock = end_total - start_total
    
    if success_res:
        avg_rtf = sum(r['rtf'] for r in success_res) / len(success_res)
        total_audio_generated = sum(r['audio_len'] for r in success_res)
        throughput = total_audio_generated / elapsed_wall_clock
        
        print(f"사용된 코어 수 : {NUM_WORKERS}개")
        print(f"성공 요청 수   : {len(success_res)} / {SIMULATED_CALLS}")
        print(f"총 소요 시간   : {elapsed_wall_clock:.2f}초")
        print(f"생성된 오디오  : {total_audio_generated:.2f}초 분량")
        print(f"처리량(Speed)  : {throughput:.2f}x (실시간 대비 배속)")
        print("-" * 40)
        print(f"평균 RTF       : {avg_rtf:.4f} (낮을수록 좋음)")
        print("-" * 40)
        
        if avg_rtf < 0.1:
            print("🏆 [Excellent] RTF 0.1 미만 달성! 상용 서비스 가능한 수준입니다.")
        elif avg_rtf < 0.3:
            print("✅ [Good] 상당히 빠릅니다. 실시간 처리에 문제 없습니다.")
        elif avg_rtf < 1.0:
            print("⚠️ [Normal] 실시간 처리는 가능하나, 동시 접속자가 많으면 밀릴 수 있습니다.")
        else:
            print("❌ [Bad] 실시간 처리가 불가능합니다.")
            
    if fail_res:
        print(f"\n❌ 실패 요청 수: {len(fail_res)}")
        print(f"에러 로그 예시: {fail_res[0]['error']}")
