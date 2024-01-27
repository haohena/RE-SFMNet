import os
from PIL import Image

input_folder = "/home/zjj/suitianbao/SFMNet/data/train/HR"
output_folder = "/home/zjj/suitianbao/SFMNet/data/train/LR_x16_bicubic"
os.makedirs(output_folder, exist_ok=True)
i= 0
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    image = Image.open(input_path)
    downsampled_image = image.resize((image.width // 16, image.height // 16), Image.BICUBIC)
    output_path = os.path.join(output_folder, filename)
    downsampled_image.save(output_path)
    i=i+1
    print(f"Saved {filename} to {output_path}")
print('共计',i,'张图片')
