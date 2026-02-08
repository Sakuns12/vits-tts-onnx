import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols
import os

# Wrapper 클래스: VITS의 여러 출력값 중 '오디오'만 뽑아냄
class VITS_ONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, text_lengths, noise_scale, length_scale, noise_scale_w):
        return self.model.infer(
            text, text_lengths, noise_scale=noise_scale,
            length_scale=length_scale, noise_scale_w=noise_scale_w
        )[0]

def main():
    # 1. 경로 설정 (본인 환경에 맞게 수정)
    model_name = 'ko_16k'
    config_file = f"./configs/{model_name}.json"
    model_file = f"./logs/{model_name}/G_0.pth"
    onnx_file = f"{model_name}.onnx"

    hps = utils.get_hparams_from_file(config_file)

    # 2. 모델 로드 (CPU 모드 권장 for Export)
    device = torch.device("cpu")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    _ = net_g.eval()
    _ = utils.load_checkpoint(model_file, net_g, None)
    
    # Wrapper 적용
    vits_onnx = VITS_ONNX(net_g)

    # 3. 더미 데이터 생성
    dummy_text = torch.randint(low=0, high=len(symbols), size=(1, 50), dtype=torch.long).to(device)
    dummy_text_lengths = torch.LongTensor([50]).to(device)
    noise_scale = torch.tensor(0.667, dtype=torch.float32).to(device)
    length_scale = torch.tensor(1.0, dtype=torch.float32).to(device)
    #noise_scale_w = torch.tensor(0.8, dtype=torch.float32).to(device)
    noise_scale_w = torch.tensor(0.1, dtype=torch.float32).to(device)

    # 4. Export 실행
    print(f"Exporting to {onnx_file}...")
    torch.onnx.register_custom_op_symbolic("aten::lift_fresh", lambda g, x: x, 13)
    torch.onnx.export(
        vits_onnx,
        (dummy_text, dummy_text_lengths, noise_scale, length_scale, noise_scale_w),
        onnx_file,
        export_params=True,
        opset_version=16,  # PyTorch 2.x에서는 16 권장
        do_constant_folding=True,
        input_names=['text', 'text_lengths', 'noise_scale', 'length_scale', 'noise_scale_w'],
        output_names=['audio'],
        dynamic_axes={
            'text': {1: 'text_length'},
            'audio': {2: 'audio_length'}
        }
    )
    print("성공! ONNX 파일 생성 완료.")

if __name__ == "__main__":
    main()
