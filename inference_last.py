import matplotlib.pyplot as plt
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import re
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import time

from scipy.io.wavfile import write

def get_text(text, hps):
    # [Tip] 8k 모델 노이즈 방지를 위해 korean_cleaners 강제 적용 권장
    # hps.data.text_cleaners 대신 ['korean_cleaners']를 직접 사용하는 것이 안전합니다.
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# 1. 설정 변수
model_name = 'ko'
config_file = f"./configs/{model_name}.json"

# [수정됨] 최신 모델 자동 로드 로직
model_dir = f"./logs/{model_name}"
try:
    # utils 라이브러리를 사용하여 G_*.pth 패턴 중 숫자가 가장 큰 파일을 찾습니다.
    model_file = utils.latest_checkpoint_path(model_dir, "G_*.pth")
    print(f"✅ 가장 최신 체크포인트를 로드합니다: {model_file}")
except Exception as e:
    print(f"⚠️ 모델을 찾을 수 없습니다. 경로를 확인하세요: {model_dir}")
    # 비상시 기본값 (필요하면 수정)
    model_file = f"./logs/{model_name}/G_0.pth"

# 2. 설정 파일 로드
hps = utils.get_hparams_from_file(config_file)

# 3. 텍스트 입력
# [Tip] 줄바꿈 문자가 있으면 jk_cleaners가 오작동할 수 있으므로 제거합니다.
text = """
    [KO]결론부터 말씀드리면, 이 방법은 비추천합니다. 
    에러 메시지는 사라지게 할 수 있지만, 만들어진 모델이 "고정된 길이"의 오디오만 처리하게 될 위험이 매우 크기 때문입니다.[KO]
    """

# 텍스트 전처리
text = re.sub('[\n]', '', text).strip()

# 4. 모델 생성 및 가중치 로드
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # VOIP 환경 가정하에 CPU로 설정 (GPU 있으면 'cuda'로 변경)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()

# 모델 로드
_ = utils.load_checkpoint(model_file, net_g, None)

# 5. 추론 (Inference)
stn_tst = get_text(text, hps)

# [디버깅] 텍스트 변환 확인
print(f"입력 텍스트 길이: {len(text)}")
print(f"변환된 토큰 개수: {stn_tst.size(0)}")

with torch.no_grad():
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    
    start_time = time.time()
    
    # noise_scale: 0.667 (기본), length_scale: 1.0 (속도 조절)
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    end_time = time.time()

print(f"추론 소요 시간: {end_time - start_time:.4f}초")

# 6. 결과 저장
output_file = "result.wav"
write(output_file, hps.data.sampling_rate, audio)
print(f"완료! '{output_file}' 파일이 생성되었습니다.")