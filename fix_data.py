import os
import glob
from tqdm import tqdm

# --- CẤU HÌNH ---
DATA_DIR = "UnsafeNet_Ready"  # Thư mục dữ liệu của bạn

def check_and_clean_dataset():
    print(f"🕵️‍♂️ Checking data health in: {DATA_DIR}...")
    
    # Lấy danh sách tất cả file label
    label_files = glob.glob(os.path.join(DATA_DIR, "labels", "**", "*.txt"), recursive=True)
    
    bad_files = 0
    
    for lbl_path in tqdm(label_files, desc="Error scanning"):
        is_bad = False
        try:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                
                # Lỗi 1: Dòng trống hoặc không đủ 5 giá trị (class x y w h)
                if len(parts) != 5:
                    is_bad = True
                    break
                
                # Lỗi 2: Không phải số
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    is_bad = True
                    break
                    
                # Lỗi 3: Tọa độ âm hoặc > 1 (Lỗi phổ biến khi tính toán sai)
                # Class ID (vals[0]) phải là số nguyên >= 0
                if vals[0] < 0 or vals[3] < 0 or vals[4] < 0: # w, h không được âm
                    is_bad = True
                    break

        except Exception:
            is_bad = True # Không đọc được file cũng là lỗi

        if is_bad:
            bad_files += 1
            # Remove error label file
            os.remove(lbl_path)
            
            # Remove file image
            # Đường dẫn ảnh: labels/train/abc.txt -> images/train/abc.jpg
            img_path = lbl_path.replace("labels", "images").replace(".txt", ".jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            
            # Try to delete all file .png or other format file
            if os.path.exists(img_path.replace(".jpg", ".png")):
                os.remove(img_path.replace(".jpg", ".png"))

    print("\n" + "="*40)
    if bad_files > 0:
        print(f"✅ Found error file and removed {bad_files} error file!")
        print("Clean data. Try to train again")
    else:
        print("✅ Health data. Error can be catched by cached")
    print("="*40)

if __name__ == "__main__":
    check_and_clean_dataset()