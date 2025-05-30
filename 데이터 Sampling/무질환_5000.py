import os
import shutil
import random

# 경로 설정
folders = [
    r"C:\Users\minar\dataset\TL1_개_안구_일반\결막염\무",
    r"C:\Users\minar\dataset\TL1_개_안구_일반\궤양성각막질환\무",
    r"C:\Users\minar\dataset\TL1_개_안구_일반\비궤양성각막질환\무",
    r"C:\Users\minar\dataset\TL1_개_안구_일반\색소침착성각막염\무",
    r"C:\Users\minar\dataset\TL2_개_안구_일반\안검내반증_무\무",
    r"C:\Users\minar\dataset\TL2_개_안구_일반\안검염_무\무",
    r"C:\Users\minar\dataset\TL2_개_안구_일반\안검종양_무\무",
    r"C:\Users\minar\dataset\TL2_개_안구_일반\유루증_유\무",
    r"C:\Users\minar\dataset\TL2_개_안구_일반\핵경화_무\무",
    r"C:\Users\minar\dataset\TL1_개_안구_일반\X_백내장\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\결막염\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\궤양성각막질환\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\비궤양성각막질환\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\색소침착성각막염\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\안검내반증\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\안검염\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\안검종양\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\유루증\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\핵경화\무",
    r"C:\Users\minar\dataset\VL_개_안구_일반\X_백내장\무"

]

target_folder = r"C:\Users\dataset\최종 데이터셋\무질환"

# 지원하는 이미지 확장자
image_extensions = ['.jpg']

# 모든 폴더에서 파일 목록 가져오기
def get_files_from_folders(folders):
    all_images = []
    all_jsons = []
    
    for folder in folders:
        files = os.listdir(folder)
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        jsons = [f for f in files if os.path.splitext(f)[1].lower() == '.json']
        
        all_images.extend(images)
        all_jsons.extend(jsons)
    
    return all_images, all_jsons

# 이미지와 JSON 매칭
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
    # 모든 폴더에서 파일 목록 가져오기
    images, jsons = get_files_from_folders(folders)

    # 이미지와 JSON 매칭
    matched_files = match_files(images, jsons)

    # 5000개 랜덤 선택 및 복사
    copy_files(matched_files, folders, target_folder, 5000)

    print(f"파일 복사가 완료되었습니다! 총 {len(matched_files)}개의 파일 중 최대 5000개가 타겟 폴더로 복사되었습니다.")
