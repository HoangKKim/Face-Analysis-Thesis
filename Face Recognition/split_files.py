# split files in a folder into different folders based on first word in filename
import os
import shutil

def split_files(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        first_word = filename.split('_')[0]
        target_folder = os.path.join(folder_path, first_word)
        
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        source_path = os.path.join(folder_path, filename)
        target_path = os.path.join(target_folder, filename)
        
        shutil.move(source_path, target_path)

if __name__ == "__main__":
    folder_path = "./Data/train"
    split_files(folder_path)