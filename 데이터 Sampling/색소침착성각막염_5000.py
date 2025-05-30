import os
import shutil
import random

# 경로 설정
folder_a = r"C:\Users\dataset\TL1_개_안구_일반\색소침착성각막염\유"
folder_b = r"C:\Users\dataset\VL_개_안구_일반\색소침착성각막염\유"
target_folder = r"C:\Users\dataset\최종 데이터셋\색소침착성각막염"

# 지원하는 이미지 확장자
image_extensions = ['.jpg']

# A와 B 폴더의 파일 목록 가져오기
def get_files(folder):
    files = os.listdir(folder)
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    jsons = [f for f in files if os.path.splitext(f)[1].lower() == '.json']
    return images, jsons

def match_files(images, jsons):
    matched = []
    for image in images:
        base_name = os.path.splitext(image)[0]
        json_file = f"{base_name}.json"
        if json_file in jsons:
            matched.append((image, json_file))
    return matched

# 랜덤 샘플링 후 파일 복사
def copy_files(matched_files, source_folders, target_folder, sample_size):
    os.makedirs(target_folder, exist_ok=True)
    
    # 샘플링
    selected_files = random.sample(matched_files, min(len(matched_files), sample_size))
    
    for image, json_file in selected_files:
        for folder in source_folders:
            image_path = os.path.join(folder, image)
            json_path = os.path.join(folder, json_file)
            
            if os.path.exists(image_path) and os.path.exists(json_path):
                shutil.copy(image_path, os.path.join(target_folder, image))
                shutil.copy(json_path, os.path.join(target_folder, json_file))
                break  # 한 번 복사했으면 다음 폴더는 확인할 필요 없음

# 실행
if __name__ == "__main__":
    # A와 B 폴더에서 파일 목록 가져오기
    images_a, jsons_a = get_files(folder_a)
    images_b, jsons_b = get_files(folder_b)

    # 이미지와 JSON 매칭
    matched_a = match_files(images_a, jsons_a)
    matched_b = match_files(images_b, jsons_b)

    # A와 B 폴더의 매칭된 파일 합치기
    all_matched = matched_a + matched_b

    # 6000개 랜덤 선택 및 복사
    copy_files(all_matched, [folder_a, folder_b], target_folder, 5000)

    print(f"파일 복사가 완료되었습니다! 총 {len(all_matched)}개의 파일 중 최대 5000개가 C 폴더로 복사되었습니다.")
