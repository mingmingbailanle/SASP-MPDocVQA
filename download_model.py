# download_model.py
import urllib.request
import os
import json

# 尝试多个镜像源
MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",  # 官方（需要代理）
]


def download_file(url, save_path):
    """尝试下载文件"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False


def download_model():
    save_dir = "models/pix2struct-base"
    os.makedirs(save_dir, exist_ok=True)

    # 必需文件列表
    files = {
        "config.json": "config.json",
        "preprocessor_config.json": "preprocessor_config.json",
        "pytorch_model.bin": "pytorch_model.bin",
        "tokenizer_config.json": "tokenizer_config.json",
        "spiece.model": "spiece.model",  # 注意文件名可能是这个
    }

    # 先尝试获取实际的文件列表
    print("正在探测可用镜像...")
    for mirror in MIRRORS:
        try:
            # 尝试读取文件列表
            api_url = f"{mirror}/api/models/google/pix2struct-base"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
                print(f"✓ 镜像可用: {mirror}")
                break
        except Exception as e:
            print(f"✗ 镜像不可用: {mirror} - {e}")
            continue
    else:
        print("所有镜像都不可用，尝试直接下载已知文件...")

    # 下载每个文件
    for filename, save_name in files.items():
        save_path = os.path.join(save_dir, save_name)

        if os.path.exists(save_path):
            print(f"✓ {save_name} 已存在")
            continue

        print(f"正在下载 {save_name} ...")

        # 尝试所有镜像
        downloaded = False
        for mirror in MIRRORS:
            # 尝试多种路径格式
            urls = [
                f"{mirror}/google/pix2struct-base/resolve/main/{filename}",
                f"{mirror}/google/pix2struct-base/resolve/main/{save_name}",
            ]

            for url in urls:
                print(f"  尝试: {url}")
                if download_file(url, save_path):
                    print(f"  ✓ 成功")
                    downloaded = True
                    break

            if downloaded:
                break

        if not downloaded:
            print(f"  ✗ 所有镜像都失败")

    print(f"\n下载完成，文件保存在: {os.path.abspath(save_dir)}")
    print("文件列表:")
    for f in os.listdir(save_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    download_model()