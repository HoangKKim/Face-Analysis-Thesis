import os
import random
import shutil

# Cấu hình
root_folder = 'output/backup_demo_fps60/students_result'
recognizer_folder = 'modules/recognizer/test_data'
os.makedirs(recognizer_folder, exist_ok=True)

n_folders_per_person = 3  # <-- Bạn thay số này tùy nhu cầu
n_images_per_folder = 50  # <-- Mỗi folder chứa 50 ảnh

label_file_path = os.path.join(recognizer_folder, 'labels.txt')
folder_counter = 1  # Đánh số folder toàn cục (liên tục)

with open(label_file_path, 'w') as label_file:
    for person in os.listdir(root_folder):
        image_folder_path = os.path.join(root_folder, person, 'face')
        if not os.path.isdir(image_folder_path):
            continue

        image_list = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_list) < n_folders_per_person * n_images_per_folder:
            print(f"⚠️ Warning: person '{person}' chỉ có {len(image_list)} ảnh. Bỏ qua.")
            continue

        for _ in range(n_folders_per_person):
            folder_name = f'folder_{folder_counter}'
            folder_path = os.path.join(recognizer_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            sampled_images = random.sample(image_list, n_images_per_folder)

            for img_name in sampled_images:
                src_path = os.path.join(image_folder_path, img_name)
                dst_path = os.path.join(folder_path, img_name)
                shutil.copy(src_path, dst_path)

            label_file.write(f"{folder_name} - {person}\n")

            folder_counter += 1
