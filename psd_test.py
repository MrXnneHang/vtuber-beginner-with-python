import numpy as np
import cv2
from psd_tools import PSDImage

def 提取图层(psd):
    所有图层 = []

    def dfs(图层, path=''):
        # 如果是图层组，则递归处理子图层
        if 图层.is_group():
            for i in 图层:
                dfs(i, path + 图层.name + '/')
        else:
            # 获取图层的边界框 (top, left, bottom, right)
            a, b, c, d = 图层.bbox

            # 将图层转换为numpy数组 (RGBA格式)
            npdata = np.array(图层.topil())

            # 交换红色和蓝色通道，因为OpenCV使用BGR格式
            npdata[:, :, 0], npdata[:, :, 2] = npdata[:, :, 2].copy(), npdata[:, :, 0].copy()

            # 保存图层信息
            所有图层.append({'名字': path + 图层.name, '位置': (b, a, d, c), 'npdata': npdata})

    # 遍历PSD中的每个图层
    for 图层 in psd:
        dfs(图层)

    return 所有图层

def 测试图层叠加(所有图层, 宽=500, 高=500):
    # 初始化一个空白图像 (使用RGBA通道)
    img = np.ones([高, 宽, 4], dtype=np.float32) * 255  # 初始为全白

    for 图层数据 in 所有图层:
        a, b, c, d = 图层数据['位置']
        新图层 = 图层数据['npdata']

        # 计算图层的实际大小
        layer_height, layer_width = 新图层.shape[:2]

        # 确保图层位置在图像范围内
        a, b = max(a, 0), max(b, 0)
        c, d = min(c, 高), min(d, 宽)

        # 重新计算叠加区域的尺寸
        target_height = c - a
        target_width = d - b

        # 计算图层应该被裁剪到的尺寸
        crop_height = min(target_height, layer_height)
        crop_width = min(target_width, layer_width)

        # 裁剪图层到目标尺寸
        新图层 = 新图层[:crop_height, :crop_width]

        # 重新计算alpha通道
        alpha = 新图层[:, :, 3] / 255.0

        # 使用alpha混合每个颜色通道
        for i in range(3):  # 对RGB三个通道进行处理
            img[a:a+crop_height, b:b+crop_width, i] = (
                img[a:a+crop_height, b:b+crop_width, i] * (1 - alpha) +
                新图层[:, :, i] * alpha
            )

        # 更新图像的alpha通道
        img[a:a+crop_height, b:b+crop_width, 3] = np.maximum(
            img[a:a+crop_height, b:b+crop_width, 3], 新图层[:, :, 3]
        )

    # 显示结果图像
    cv2.imshow('叠加后的图像', img.astype(np.uint8))  # 转换为8位整数显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 加载PSD文件
    psd = PSDImage.open('./res/test.psd')  # 将example.psd替换为你的PSD文件路径

    # 提取PSD图层
    所有图层 = 提取图层(psd)

    # 测试图层叠加
    测试图层叠加(所有图层, 宽=psd.width, 高=psd.height)
