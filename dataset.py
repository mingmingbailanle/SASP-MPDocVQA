import os

# 设置 Hugging Face 镜像站点，加速模型下载（国内必需）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from transformers import Pix2StructProcessor

IMDB = "./dataset/imdb/"
IMG = "./dataset/images/"


class MPDocVQA(Dataset):
    def __init__(self, imdb_dir, img_dir, split, processor=None, model_name='google/pix2struct-base'):
        """
        参数:
            imdb_dir: IMDB 数据文件路径字典
            img_dir: 图像目录路径
            split: 数据集划分 ('train', 'val', 'test')
            processor: 可选，外部传入的处理器（避免重复加载）
            model_name: 模型名称，默认 'google/pix2struct-base'
        """
        data = np.load(imdb_dir[split], allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]
        self.img_dir = img_dir
        self.split = split  # 保存split用于调试

        # 加载处理器：优先使用外部传入的，否则自行加载
        if processor is not None:
            self.processor = processor
            print(f"[{split}] Using provided processor")
        else:
            # 修复：使用正确的模型 ID（原 'google/pix2struct-docvqa-base' 不存在）
            print(f"[{split}] Loading processor from: {model_name}")
            self.processor = Pix2StructProcessor.from_pretrained(
                model_name,  # 改为 'google/pix2struct-base'
                use_fast=False
            )

    def __len__(self):
        return len(self.imdb)

    def get_random_item(self, except_idx):
        false_idx = random.choice([x for x in range(self.__len__()) if x != except_idx])
        return self.imdb[false_idx]

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        ques_id = record['question_id']
        answers = list(set(answer.lower() for answer in record['answers']))
        image_names = record['image_name']
        answer_page_idx = record['answer_page_idx']
        doc_pages = record['imdb_doc_pages']

        # 采样正负样本
        if doc_pages > 1:
            false_idx = random.choice([x for x in range(doc_pages) if x != answer_page_idx])
            final_image_names = [image_names[answer_page_idx], image_names[false_idx]]
            final_rela_probs = [random.uniform(0.88, 1.), random.uniform(0., 0.12)]
        else:
            final_image_names = [image_names[answer_page_idx], image_names[answer_page_idx]]
            final_rela_probs = [random.uniform(0.88, 1.), random.uniform(0.88, 1.)]

        final_patches = []
        final_masks = []

        for img_name in final_image_names:
            # 构建完整图像路径
            img_path = os.path.join(self.img_dir, f'{img_name}.jpg')

            # 调试：检查文件是否存在（首次运行时）
            # if idx == 0:
            #     print(f"尝试加载图像: {img_path}")
            #     print(f"文件存在: {os.path.exists(img_path)}")

            try:
                image = Image.open(img_path).convert('RGB')
            except (OSError, UnidentifiedImageError, FileNotFoundError) as e:
                print(f"[{self.split}] Skip corrupted/missing image at idx={idx}: {img_path} ({e})")
                return None

            # 使用 processor 处理图像和文本
            inputs = self.processor(
                images=[image],
                text=[question],
                return_tensors='np',
                padding=True
            )

            flattened_patches = inputs.flattened_patches
            patches_padding_mask = inputs.attention_mask
            final_patches.append(flattened_patches.squeeze(0))
            final_masks.append(patches_padding_mask.squeeze(0))

        sample_info = {
            'question_id': ques_id,
            'question': question,
            'answers': answers,
            'image_names': final_image_names,
            'image_patches': final_patches,
            'patches_masks': final_masks,
            'rela_probs': final_rela_probs,
            'answer_page_idx': answer_page_idx,
            'num_pages': doc_pages
        }

        return sample_info


def loadData(processor=None, model_name='google/pix2struct-base'):
    """
    加载数据集

    参数:
        processor: 可选，共享的处理器实例
        model_name: 模型名称，用于创建处理器

    返回:
        data_train, data_val: 训练和验证数据集
    """
    imdb_base = IMDB
    img_dir = IMG
    imdb_dir = dict()

    for split in ['train', 'val', 'test']:
        imdb_dir[split] = os.path.join(imdb_base, f'imdb_{split}.npy')

    # 创建数据集：传入共享的 processor 避免重复加载
    data_train = MPDocVQA(imdb_dir, img_dir, split='train', processor=processor, model_name=model_name)
    data_val = MPDocVQA(imdb_dir, img_dir, split='val', processor=processor, model_name=model_name)

    return data_train, data_val


if __name__ == '__main__':
    # 测试配置
    imdb_base = "./dataset/imdb/"
    img_base = "./dataset/images/"

    imdb_dir = {
        'train': os.path.join(imdb_base, 'imdb_train.npy'),
        'val': os.path.join(imdb_base, 'imdb_val.npy'),
        'test': os.path.join(imdb_base, 'imdb_test.npy')
    }

    # 关键：传入字符串路径
    mp_docvqa = MPDocVQA(imdb_dir, img_base, split='val')

    # 测试
    print(f"数据集大小: {len(mp_docvqa)}")
    sample = mp_docvqa.__getitem__(0)  # 从0开始测试
    print(f"Sample keys: {sample.keys()}")
    print(f"Question: {sample['question']}")
    print(f"Answers: {sample['answers']}")
    print(f"Image names: {sample['image_names']}")
    print(f"Patches shape: {[p.shape for p in sample['image_patches']]}")