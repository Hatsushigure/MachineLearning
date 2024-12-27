import sys
import torch
from datetime import datetime
import uuid


def get_mac():
    mac = uuid.getnode()
    mac_address = ':'.join(('%012X' % mac)[i:i + 2] for i in range(0, 12, 2))
    return mac_address


def main():
    if len(sys.argv) != 3:
        print("用法: python ml.py <姓名> <学号>")
        sys.exit(1)

    name = sys.argv[1]
    student_id = sys.argv[2]

    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # 打印信息
    print(f"姓名: {name}, 学号: {student_id}, 当前时间: {formatted_time}")
    print(f"PyTorch 版本: {torch.__version__}, MAC: {get_mac()}")


if __name__ == "__main__":
    main()