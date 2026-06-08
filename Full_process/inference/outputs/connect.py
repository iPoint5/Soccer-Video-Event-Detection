from PIL import Image

# 读取四张图片
img1 = Image.open(r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\Shots on target.png")
img2 = Image.open(r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\Shots on target.png")
img3 = Image.open(r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\Corner.png")
img4 = Image.open(r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\Clearance.png")

# 获取单张图片尺寸
w, h = img1.size

# 创建画布（2×2）
canvas = Image.new("RGB", (2 * w, 2 * h), "white")

# 拼接
canvas.paste(img1, (0, 0))
canvas.paste(img2, (w, 0))
canvas.paste(img3, (0, h))
canvas.paste(img4, (w, h))

# 保存
canvas.save("merged.png")

print("保存成功：merged.png")