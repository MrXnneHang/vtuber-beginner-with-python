import numpy as np
import cv2
from psd_tools import PSDImage

def extract_layers(psd):
    """
    提取PSD文件中的图层并返回图层信息列表。

    参数:
        psd (PSDImage): 表示PSD文件的PSDImage对象。

    返回:
        list: 一个包含每个图层信息的字典列表，其中包括以下键:
            - 'name' (str): 图层的完整路径名称。
            - 'position' (tuple): 图层的边界框，格式为(left, top, right, bottom)。
            - 'npdata' (np.ndarray): 图层图像数据，格式为NumPy数组(BGRA格式)。
    """
    all_layers = []

    def dfs(layer, path=''):
        # 递归遍历图层树
        if layer.is_group():
            for sub_layer in layer:
                dfs(sub_layer, path + layer.name + '/')
        else:
            # 获取图层的边界框 (top, left, bottom, right)
            a, b, c, d = layer.bbox

            # 将图层转换为NumPy数组 (RGBA格式)
            npdata = np.array(layer.topil())

            # 交换红色和蓝色通道，因为OpenCV使用BGR格式
            npdata[:, :, 0], npdata[:, :, 2] = npdata[:, :, 2].copy(), npdata[:, :, 0].copy()

            # 保存图层信息
            all_layers.append({'name': path + layer.name, 'position': (b, a, d, c), 'npdata': npdata})

    # 遍历PSD中的每个顶层图层
    for layer in psd:
        dfs(layer)

    return all_layers

def test_layer_composition(all_layers, width=500, height=500):
    """
    测试图层的叠加效果，通过将每个图层按顺序叠加到一个空白图像上进行可视化。

    参数:
        all_layers (list): 从PSD文件中提取的图层信息列表。
        width (int): 输出图像的宽度，默认为500。
        height (int): 输出图像的高度，默认为500。

    返回:
        None: 显示合成后的图像。
    """
    # 初始化一个空白图像 (使用RGBA通道)
    img = np.ones([height, width, 4], dtype=np.float32) * 255  # 初始为全白

    for layer_data in all_layers:
        a, b, c, d = layer_data['position']
        new_layer = layer_data['npdata']

        # 计算图层的实际大小
        layer_height, layer_width = new_layer.shape[:2]

        # 确保图层位置在图像范围内
        a, b = max(a, 0), max(b, 0)
        c, d = min(c, height), min(d, width)

        # 重新计算叠加区域的尺寸
        target_height = c - a
        target_width = d - b

        # 计算图层应该被裁剪到的尺寸
        crop_height = min(target_height, layer_height)
        crop_width = min(target_width, layer_width)

        # 裁剪图层到目标尺寸
        new_layer = new_layer[:crop_height, :crop_width]

        # 重新计算alpha通道
        alpha = new_layer[:, :, 3] / 255.0

        # 使用alpha混合每个颜色通道
        for i in range(3):  # 对RGB三个通道进行处理
            img[a:a+crop_height, b:b+crop_width, i] = (
                img[a:a+crop_height, b:b+crop_width, i] * (1 - alpha) +
                new_layer[:, :, i] * alpha
            )

        # 更新图像的alpha通道
        img[a:a+crop_height, b:b+crop_width, 3] = np.maximum(
            img[a:a+crop_height, b:b+crop_width, 3], new_layer[:, :, 3]
        )

    # 显示结果图像
    cv2.imshow('叠加后的图像', img.astype(np.uint8))  # 转换为8位整数显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
