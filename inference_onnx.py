import onnxruntime as ort
import numpy as np
import utils
import commons
from text import text_to_sequence
import re
import time
from scipy.io.wavfile import write
import os

# 필요시 주석 해제 (프로젝트 구조에 따라 다름)
from text.k2j import korean2katakana
from text.j2k import japanese2korean

def get_text(text, hps):
    # 1. Text Cleaner (config.json에 설정된 cleaners 사용)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    
    # 2. Add Blank (VITS 학습 시 사용된 설정)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
        
    text_norm = np.array(text_norm, dtype=np.int64)
    return text_norm

def main():
    # -----------------------------------------
    # [설정] 경로 및 텍스트 지정
    # -----------------------------------------
    model_name = 'ko_16k'
    config_file = f"./configs/ko_16k.json"
    onnx_file = f"{model_name}.onnx"
    output_wav = "result_onnx.wav"
    
    # 테스트할 텍스트
    text = """
    [KO]결론부터 말씀드리면, 이 방법은 비추천합니다. 
    에러 메시지는 사라지게 할 수 있지만, 만들어진 모델이 "고정된 길이"의 오디오만 처리하게 될 위험이 매우 크기 때문입니다.[KO]
    """

    # 하이퍼파라미터 로드
    hps = utils.get_hparams_from_file(config_file)

    # -----------------------------------------
    # 1. 텍스트 전처리 (G2P & Cleaning)
    # -----------------------------------------
    text = re.sub('[\n]', '', text).strip()
    
    # 한국어/일본어 태그 처리 (사용 환경에 맞게 활성/비활성)
    if hasattr(hps.data, 'is_japanese_dataset') and hps.data.is_japanese_dataset:
        text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean2katakana(x.group(1)), text)
    elif hasattr(hps.data, 'is_korean_dataset') and hps.data.is_korean_dataset:
        text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese2korean(x.group(1)), text)

    # 텍스트 -> 시퀀스(ID) 변환
    stn_tst = get_text(text, hps)
    
    # ONNX 입력 포맷에 맞게 차원 확장 (Batch Dimension 추가)
    x_tst = np.expand_dims(stn_tst, axis=0)        # [1, Length]
    x_tst_lengths = np.array([x_tst.shape[1]], dtype=np.int64) # [1]

    # -----------------------------------------
    # 2. ONNX 런타임 세션 로드
    # -----------------------------------------
    # CPU 사용 설정 (RTF 0.1 목표)
    sess_options = ort.SessionOptions()
    # 쓰레드 수 조절이 필요하면 아래 주석 해제 (보통 기본값이 최적)
    # sess_options.intra_op_num_threads = 4 
    
    print(f"Loading {onnx_file}...")
    ort_session = ort.InferenceSession(onnx_file, sess_options, providers=['CPUExecutionProvider'])

    # 입력 딕셔너리 준비 (모든 입력은 numpy array여야 함)
    inputs = {
        'text': x_tst,
        'text_lengths': x_tst_lengths,
        'noise_scale': np.array([0.667], dtype=np.float32),
        'length_scale': np.array([1.0], dtype=np.float32)
    }

    # -----------------------------------------
    # 3. 추론 및 RTF 측정
    # -----------------------------------------
    print("Inference starting...")
    start_time = time.time()
    
    # ONNX 실행
    audio = ort_session.run(None, inputs)[0]
    
    end_time = time.time()
    inference_time = end_time - start_time

    # -----------------------------------------
    # 4. 결과 처리 (차원 축소 & 정규화)
    # -----------------------------------------
    audio = audio.squeeze() # [Batch, 1, Time] -> [Time]
    
    # RTF 계산
    audio_duration = len(audio) / hps.data.sampling_rate
    rtf = inference_time / audio_duration
    
    print("="*30)
    print(f"생성된 오디오 길이 : {audio_duration:.2f}s")
    print(f"추론 소요 시간     : {inference_time:.4f}s")
    print(f"RTF (Real Time Factor): {rtf:.4f}")
    print("="*30)

    # 안전한 정규화 (Peak Normalization)
    max_val = np.abs(audio).max()
    if max_val > 0.99:
        print(f"Warning: Audio clipping detected (Max: {max_val:.2f}). Normalizing...")
        audio = audio / max_val * 0.95
    
    # -----------------------------------------
    # 5. 파일 저장
    # -----------------------------------------
    write(output_wav, hps.data.sampling_rate, audio)
    print(f"저장 완료: {output_wav}")

if __name__ == "__main__":
    main()
