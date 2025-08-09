# split_train_val.py  放在 binocular_yolov5 根目录下执行:  python split_train_val.py
import random, shutil
from pathlib import Path

root   = Path('ai_game_data')
trainV = root/'train/vis'    # 可见光
trainI = root/'train/ir'     # 红外
trainL = root/'train/label'  # 标签

valV   = root/'val/vis'
valI   = root/'val/ir'
valL   = root/'val/label'

# 1. 创建验证集目录
for p in (valV, valI, valL):
    p.mkdir(parents=True, exist_ok=True)

# 2. 统计全部样本（以可见光 jpg/png 为准）
imgs = sorted([p for p in trainV.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
assert imgs, f'未找到图片，请确认路径 {trainV}'

# 3. 随机抽 20% 作为验证集
k = len(imgs) // 5          # 2/10 = 20%
val_imgs = set(random.sample(imgs, k))
print(f'共 {len(imgs)} 张图像，抽出 {k} 张做验证集')

# 4. 逐文件迁移（vis、ir、label 三者同名）
for vis_img in val_imgs:
    stem = vis_img.stem
    ir_img  = trainI / f'{stem}{vis_img.suffix}'
    label_f = trainL / f'{stem}.txt'

    # 移动到 val
    shutil.move(str(vis_img), valV / vis_img.name)
    if ir_img.exists():
        shutil.move(str(ir_img), valI / ir_img.name)
    if label_f.exists():
        shutil.move(str(label_f), valL / label_f.name)

print('完成数据集拆分 ✔')