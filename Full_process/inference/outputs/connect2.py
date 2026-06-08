import os

folder_path = r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs"  # 修改为你的图片文件夹路径

INVERSE_EVENT_DICTIONARY_V2 = {
    0: "Penalty",
    1: "Kick-off",
    2: "Goal",
    3: "Substitution",
    4: "Offside",
    5: "Shots on target",
    6: "Shots off target",
    7: "Clearance",
    8: "Ball out of play",
    9: "Throw-in",
    10: "Foul",
    11: "Indirect free-kick",
    12: "Direct free-kick",
    13: "Corner",
    14: "Yellow card",
    15: "Red card",
    16: "Yellow->red card"
}

for idx, event_name in INVERSE_EVENT_DICTIONARY_V2.items():
    old_file = os.path.join(folder_path, f"{idx}.png")

    # Windows 文件名不能包含这些字符
    safe_name = event_name.replace("/", "-").replace(":", "-")
    new_file = os.path.join(folder_path, f"{safe_name}.png")

    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"{idx}.png -> {safe_name}.png")
    else:
        print(f"未找到: {old_file}")

print("重命名完成")