import os

# 1. 설정: 파일 경로를 본인 환경에 맞게 수정하세요.
input_file = "transcript.v.1.4.txt"  # KSS 원본 스크립트 파일명
output_file = "ko_train.txt"  # 저장할 파일명

def convert_transcript():
    print(f"📂 '{input_file}' 변환을 시작합니다...")
    
    converted_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split('|')
            
            # KSS 데이터 형식: 
            # 1/1_0000.wav|그는...|그는...|그는...|3.5|He...
            if len(parts) >= 2:
                # 1. 파일명 추출 (1/1_0000.wav -> 1_0000.wav)
                original_path = parts[0]
                filename = os.path.basename(original_path) # 경로 떼고 파일명만
                
                # 2. 텍스트 추출 (두 번째 항목)
                text = parts[1]
                
                # 3. 새로운 포맷으로 조합
                # dataset/KO/1_0000.wav|[KO]그는 괜찮은 척하려고 애쓰는 것 같았다.[KO]
                new_line = f"dataset/KO/{filename}|[KO]{text}[KO]"
                converted_lines.append(new_line)

        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(converted_lines))
            
        print(f"✅ 변환 완료! 총 {len(converted_lines)}개의 라인이 저장되었습니다.")
        print(f"📄 저장된 파일: {output_file}")
        
        # 미리보기 (처음 3줄)
        print("\n--- [결과 미리보기] ---")
        for i in range(min(3, len(converted_lines))):
            print(converted_lines[i])
            
    except FileNotFoundError:
        print(f"❌ 에러: '{input_file}' 파일을 찾을 수 없습니다. 파일명을 확인해주세요.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    convert_transcript()