from psd_tools import PSDImage
from utils.psd_layer_util import extract_layers, test_layer_composition

if __name__ == "__main__":
    # 加载PSD文件
    psd = PSDImage.open('./res/test.psd')  # 将example.psd替换为你的PSD文件路径

    # 提取PSD图层
    all_layers = extract_layers(psd)

    # 测试图层叠加
    test_layer_composition(all_layers, width=psd.width, height=psd.height,scale_factor=0.7)
