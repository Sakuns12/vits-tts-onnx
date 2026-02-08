import os
import glob
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ==========================================
# 🛠️ 설정 (200ms 패딩 적용)
# ==========================================
INPUT_DIR = "./"             # 원본 wav 파일들이 있는 경로 (현재 경로에 wav가 있다면)
OUTPUT_DIR = "./dataset/KO"  # 결과물이 저장될 경로 (학습에 바로 쓸 수 있게 폴더 지정)
SAMPLE_RATE = 16000          # 목표 샘플링 레이트 (16k)
TOP_DB = 30                  # 묵음 감지 기준 (30dB)
PAD_MS = 200                 # ✅ 요청하신 200ms (0.2초) 여백 설정
# ==========================================

def trim_silence_with_padding():
    # 결과 폴더가 없으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 현재 폴더의 모든 wav 파일 찾기 (하위 폴더 포함하고 싶으면 recursive=True 사용 필요)
    # 여기서는 단순하게 현재 폴더의 wav만 찾습니다.
    # 만약 kss 구조(1/*.wav)대로라면 glob 패턴을 수정해야 합니다.
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    # 파일이 없다면 재귀적으로 찾기 시도 (혹시 모르니)
    if not wav_files:
        wav_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.wav"), recursive=True)

    print(f"🔍 총 {len(wav_files)}개의 파일을 처리합니다. (Pad: {PAD_MS}ms)")

    for wav_path in tqdm(wav_files):
        try:
            # 1. 오디오 로드 및 16k 리샘플링
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
            
            # 2. 묵음 제거 (Trimming)
            # 30dB 이하의 소리를 앞뒤로 잘라냅니다.
            y_trimmed, index = librosa.effects.trim(y, top_db=TOP_DB)
            
            # 3. 200ms 패딩 생성
            # 샘플 수 = 16000 * 0.2 = 3200개 샘플
            pad_len = int(SAMPLE_RATE * (PAD_MS / 1000))
            padding = np.zeros(pad_len)
            
            # 4. [패딩 + 오디오 + 패딩] 합치기
            y_final = np.concatenate([padding, y_trimmed, padding])
            
            # 5. 저장
            filename = os.path.basename(wav_path)
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            sf.write(save_path, y_final, SAMPLE_RATE, subtype='PCM_16')
            
        except Exception as e:
            print(f"❌ 건너뜀 ({wav_path}): {e}")

    print(f"\n✅ 완료! '{OUTPUT_DIR}' 폴더에 200ms 여백이 적용된 파일들이 저장되었습니다.")
    print(f"👉 이제 filelist.txt 경로를 '{OUTPUT_DIR}/파일명.wav' 형식으로 맞춰주세요.")

if __name__ == "__main__":
    trim_silence_with_padding()