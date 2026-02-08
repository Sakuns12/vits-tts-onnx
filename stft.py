import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window

class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        
        # [1. VITS 학습시 사용된 pinv 방식 기저 벡터 생성] (음질 떨림 방지 핵심)
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        
        # np.linalg.pinv를 사용하여 역변환 행렬 계산
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )
        
        if window is not None:
            assert(filter_length >= win_length)
            fft_window = get_window(window, win_length, fftbins=True)
            
            # Pad Center 구현
            n = filter_length
            l = fft_window.shape[0]
            if n > l:
                padding = [(n - l) // 2, (n - l) - (n - l) // 2]
                fft_window = np.pad(fft_window, padding, mode='constant')
            
            fft_window = torch.from_numpy(fft_window).float()
            
            # Basis에 윈도우 적용
            forward_basis *= fft_window
            inverse_basis *= fft_window
            
            # [Window Sum 계산용 커널]
            window_sq = fft_window ** 2
            self.register_buffer('window_sq', window_sq.view(1, 1, -1))

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)
        
        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)
        
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)
        
        return magnitude, phase

    def inverse(self, magnitude, phase):
        # 1. 복소수 재결합 (ONNX 호환: 실수 연산)
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)
        
        # 2. 역변환 (ONNX 호환: ConvTranspose1d)
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)
        
        if self.window is not None:
            # 3. Window Sum 보정 (ONNX 'lift_fresh' 에러 해결)
            
            # [핵심 수정] torch.ones(n_frames) 대신 torch.ones_like 사용
            # n_frames라는 스칼라 값을 쓰지 않고, 입력 텐서(magnitude)의 형상을 그대로 본뜹니다.
            # magnitude: [Batch, Freq, Time] -> slicing -> [Batch, 1, Time]
            ones = torch.ones_like(magnitude[:, :1, :])
            
            window_sum = F.conv_transpose1d(
                ones,
                self.window_sq,
                stride=self.hop_length,
                padding=0
            )

            # 길이 보정: ConvTranspose1d 결과가 미세하게 다를 수 있어 잘라줍니다.
            output_len = inverse_transform.size(2)
            if window_sum.size(2) > output_len:
                window_sum = window_sum[:, :, :output_len]
            elif window_sum.size(2) < output_len:
                # 매우 드물지만 짧을 경우를 대비한 패딩
                window_sum = F.pad(window_sum, (0, output_len - window_sum.size(2)))
            
            # 4. Normalization (Safe Division)
            # tiny 값 하드코딩 (1e-8) 대신 안전한 마스킹
            mask = window_sum > 1e-8
            window_sum_safe = torch.where(mask, window_sum, torch.ones_like(window_sum))
            
            inverse_transform = inverse_transform / window_sum_safe
            
            # 5. Scale Factor (VITS 학습 코드와 스케일 맞추기)
            inverse_transform = inverse_transform * (float(self.filter_length) / self.hop_length)
            
            # 6. Center Padding 제거
            pad_amt = int(self.filter_length / 2)
            inverse_transform = inverse_transform[:, :, pad_amt:-pad_amt]

        return inverse_transform.unsqueeze(1)

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction