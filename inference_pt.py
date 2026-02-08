import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import re
#from text.k2j import korean2katakana
#from text.j2k import japanese2korean
from scipy.io.wavfile import write
import os
import numpy as np
import time  # [추가] 시간 측정을 위해 필요

def main():
    # 1. 설정 변수
    model_name = 'ko_16k'
    config_file = f"./configs/{model_name}.json"
    model_file = f"./logs/{model_name}/G_0.pth"
    output_wav = "result.wav"  # 결과가 저장될 파일명
    
    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 모델 및 설정 로드
    if not os.path.exists(config_file) or not os.path.exists(model_file):
        print("오류: config 파일이나 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    hps = utils.get_hparams_from_file(config_file)
    isJaModel = hps.data.is_japanese_dataset
    isKoModel = hps.data.is_korean_dataset

    # 3. 텍스트 입력
    text = """
    [KO]결론부터 말씀드리면, 이 방법은 비추천합니다. 
    에러 메시지는 사라지게 할 수 있지만, 만들어진 모델이 "고정된 길이"의 오디오만 처리하게 될 위험이 매우 크기 때문입니다.[KO]
    """

    # 텍스트 전처리
    text = re.sub('[\n]', '', text).strip()
    '''
    if isJaModel:
        text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean2katakana(x.group(1)), text)
    if isKoModel:
        text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese2korean(x.group(1)), text)
    '''
    print(f"Processing Text: {text}")

    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    # 4. 모델 생성 및 가중치 로드
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_file, net_g, None)

    # 5. 추론 (Inference) 및 RTF 측정
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        
        # [수정] 타이머 시작
        start_time = time.time()
        
        # 추론 실행
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

        # [수정] 타이머 종료
        end_time = time.time()
        
        # 순수 추론 시간 계산
        inference_time = end_time - start_time
        
        # 생성된 오디오 길이 계산 (초 단위)
        audio_duration = len(audio) / hps.data.sampling_rate
        
        # RTF 계산 (처리 시간 / 오디오 길이)
        rtf = inference_time / audio_duration

        # 결과 출력
        print("="*30)
        print(f"생성된 오디오 길이 : {audio_duration:.4f}s")
        print(f"추론 소요 시간     : {inference_time:.4f}s")
        print(f"RTF (Real Time Factor): {rtf:.4f}")
        print("="*30)

        # 1. 현재 오디오의 최대 절댓값(피크)을 찾습니다.
        max_val = np.abs(audio).max()
        
        # 2. 피크가 1.0을 넘거나 너무 작으면 정규화
        if max_val > 1.0:
           audio = audio / max_val * 0.95

    # 6. Wav 파일 저장
    write(output_wav, hps.data.sampling_rate, audio)
    print(f"완료! 결과가 '{output_wav}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()