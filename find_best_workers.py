import time
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import onnxruntime as ort
import re

# VITS 모듈 임포트
import utils
import commons
from text import text_to_sequence
from text.k2j import korean2katakana
from text.j2k import japanese2korean

# ==========================================
# [설정]
# ==========================================
MODEL_PATH = "ko.onnx"       # 양자화 모델이 있다면 ko_quant.onnx 로 변경 권장
CONFIG_PATH = "./configs/ko.json"
TEST_TEXT = "[KO]RTF 최적화를 위한 워커 수 별 성능 측정 테스트입니다.[KO]"

# 테스트할 워커 수 목록 (서버 사양에 맞춰 자동 설정됨)
# 예: [1, 2, 4, 8, 12, 16, ...]
cpu_count = os.cpu_count()
WORKER_CANDIDATES = sorted(list(set([1, 2, 4, 8, 12, 16, 24, 32, cpu_count, cpu_count-2])))
WORKER_CANDIDATES = [w for w in WORKER_CANDIDATES if w <= cpu_count and w > 0]

CALLS_PER_WORKER = 5  # 각 워커당 처리할 요청 수 (총 요청 = 워커 수 * 이 값)

# 전역 변수
ort_session = None
hps = None

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = np.array(text_norm, dtype=np.int64)
    return text_norm

def init_worker():
    global ort_session, hps
    try:
        # 스레드 경합 방지 (매우 중요)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        hps = utils.get_hparams_from_file(CONFIG_PATH)
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        ort_session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    except:
        pass

def run_inference(dummy_idx):
    global ort_session, hps
    try:
        # 전처리
        text_proc = re.sub('[\n]', '', TEST_TEXT).strip()
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
        }

        # 추론
        t1 = time.time()
        audio = ort_session.run(None, inputs)[0]
        t2 = time.time()
        
        # 오디오 길이
        audio = audio.squeeze()
        duration = len(audio) / hps.data.sampling_rate
        
        return t2 - t1, duration # 소요시간, 오디오길이
    except Exception as e:
        return None

def benchmark(num_workers):
    print(f"\n>>> 테스트 중: 워커 {num_workers}개 ...", end="", flush=True)
    
    total_calls = num_workers * CALLS_PER_WORKER
    executor = ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker)
    
    # 웜업
    warmups = [executor.submit(run_inference, -1) for _ in range(num_workers)]
    for f in warmups: f.result()
    
    # 실제 측정
    start_total = time.time()
    futures = [executor.submit(run_inference, i) for i in range(total_calls)]
    results = [f.result() for f in futures]
    end_total = time.time()
    
    executor.shutdown()
    
    valid_results = [r for r in results if r is not None]
    if not valid_results: return 0, 0, 0

    inference_times = [r[0] for r in valid_results]
    audio_durations = [r[1] for r in valid_results]
    
    avg_rtf = (sum(inference_times) / len(inference_times)) / (sum(audio_durations) / len(audio_durations))
    elapsed = end_total - start_total
    total_audio = sum(audio_durations)
    throughput = total_audio / elapsed # 초당 생성 가능한 오디오 시간 (배속)
    
    print(f" 완료! (RTF: {avg_rtf:.4f} / 처리량: {throughput:.2f}배속)")
    return avg_rtf, throughput

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except: pass
    
    print(f"=== 최적의 워커 수 찾기 (CPU: {os.cpu_count()} cores) ===")
    print(f"모델: {MODEL_PATH}")
    
    best_throughput = 0
    best_worker_tp = 0
    
    results = []
    
    for w in WORKER_CANDIDATES:
        rtf, throughput = benchmark(w)
        results.append((w, rtf, throughput))
        
        if throughput > best_throughput:
            best_throughput = throughput
            best_worker_tp = w
            
    print("\n\n=== 📊 최종 결과 리포트 ===")
    print(f"{'워커 수':<10} | {'평균 RTF (응답속도)':<20} | {'처리량 (동시처리력)':<20}")
    print("-" * 60)
    for w, rtf, tp in results:
        mark = "⭐ (Best)" if w == best_worker_tp else ""
        # RTF가 0.15 이하이면서 처리량이 높은 구간이 실사용에 가장 좋습니다.
        quality = "쾌적" if rtf < 0.15 else ("보통" if rtf < 0.5 else "느림")
        print(f"{w:<10} | {rtf:.4f} ({quality})      | {tp:.2f}x {mark}")
    
    print("-" * 60)
    print(f"✅ 추천 설정: NUM_WORKERS = {best_worker_tp}")
    print("   (이 설정이 서버 자원을 최대로 활용하여 가장 많은 오디오를 생성합니다)")
    print("   만약 '개별 응답 속도'가 더 중요하다면 RTF < 0.1 인 워커 수를 선택하세요.")