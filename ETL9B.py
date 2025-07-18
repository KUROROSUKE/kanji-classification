"""
import struct
from PIL import Image
import os
from tqdm import tqdm

nrecords = 121440
record_size = 576
count_dic = {}
for root, dirs, files in os.walk('ETL9B'):
    for file in sorted(files):
        if file == 'ETL9INFO':
            continue
        filepath = os.path.join(root, file)
        print(file)

        with open(filepath, 'rb') as f:
            for rc_idx in tqdm(range(nrecords)):
                f.seek((rc_idx + 1) * record_size)
                s = f.read(record_size)
                r = struct.unpack('>2H4s504s64x', s)
                im = Image.frombytes('1', (64, 63), r[3], 'raw')

                code = r[1]
                idx = 0 if not (code in count_dic) else count_dic[code]
                count_dic[code] = idx + 1

                dir_name = f"./ETL9B/etl9b_chars/{format(idx, '04d')}"
                os.makedirs(dir_name, exist_ok=True)
                im.save(os.path.join(dir_name, f"img{format(code, '04x')}.png"), 'PNG')
"""

# 2個のファイルにまとめるのは何となくいやだからこうする
# 漢字ごとにまとめる。
import os
import shutil

# 変換元フォルダ
src_root = "ETL9B/etl9b_chars"
# 出力先フォルダ
dst_root = "chars"
os.makedirs(dst_root, exist_ok=True)

# フォルダをスキャン
for subdir in sorted(os.listdir(src_root)):
    subpath = os.path.join(src_root, subdir)
    if not os.path.isdir(subpath):
        continue
    for fname in os.listdir(subpath):
        if not fname.endswith(".png"):
            continue
        code = fname[3:7].lower()  # img3a2a.png → "3a2a"
        dst_dir = os.path.join(dst_root, code)
        os.makedirs(dst_dir, exist_ok=True)
        src_file = os.path.join(subpath, fname)
        # 重複を避けるため連番に変更
        new_fname = f"img_{subdir}_{fname[-8:]}"
        dst_file = os.path.join(dst_dir, new_fname)
        shutil.copyfile(src_file, dst_file)

print("✅ コード別フォルダへの整理が完了しました！")
