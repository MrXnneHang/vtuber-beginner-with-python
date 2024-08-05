import cv2
import dlib
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 初始化人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

def get_largest_face_location(image):
    """
    检测图像中的最大人脸。

    参数:
        image (numpy.ndarray): 输入图像。

    返回:
        dlib.rectangle 或 None: 检测到的最大人脸的位置，如果未检测到人脸则返回 None。
    """
    faces = detector(image, 0)
    if not faces:
        return None
    # 返回面积最大的脸的位置
    return max(faces, key=lambda face: (face.right() - face.left()) * (face.bottom() - face.top()))

def extract_facial_landmarks(image, face_location):
    """
    提取图像中人脸的关键点。

    参数:
        image (numpy.ndarray): 输入图像。
        face_location (dlib.rectangle): 图像中人脸的位置。

    返回:
        list[numpy.ndarray]: 关键点的 (x, y) 坐标列表。
    """
    landmarks = predictor(image, face_location)
    keypoints = []
    for i in range(68):
        pos = landmarks.part(i)
        keypoints.append(np.array([pos.x, pos.y], dtype=np.float32))
    return keypoints

def calculate_face_features(keypoints):
    """
    计算面部的旋转特征量（左右和上下旋转量）。

    参数:
        keypoints (list[numpy.ndarray]): 面部关键点坐标列表。

    返回:
        numpy.ndarray: (横旋转量, 竖旋转量)的数组。
    """
    def center(indexes):
        return np.mean([keypoints[i] for i in indexes], axis=0)
    
    left_eyebrow = [18, 19, 20, 21]
    right_eyebrow = [22, 23, 24, 25]
    chin = [6, 7, 8, 9, 10]
    nose = [29, 30]

    eyebrow_center = center(left_eyebrow + right_eyebrow)
    chin_center = center(chin)
    nose_center = center(nose)

    # 将二维向量扩展为三维向量
    midline = np.array([eyebrow_center[0] - chin_center[0], eyebrow_center[1] - chin_center[1], 0])
    side = np.array([eyebrow_center[0] - nose_center[0], eyebrow_center[1] - nose_center[1], 0])

    horizontal_rotation = np.cross(midline, side)[2] / np.linalg.norm(midline)**2
    vertical_rotation = np.dot(midline, side) / np.linalg.norm(midline)**2

    return np.array([horizontal_rotation, vertical_rotation])



def draw_vtuber_face(horizontal_rotation, vertical_rotation):
    """
    根据面部旋转特征量绘制简单的Vtuber脸。

    参数:
        horizontal_rotation (float): 横向旋转量。
        vertical_rotation (float): 纵向旋转量。

    返回:
        numpy.ndarray: 绘制的图像。
    """
    img = np.ones([512, 512], dtype=np.float32)
    face_length = 200
    center = (256, 256)
    left_eye = (int(220 + horizontal_rotation * face_length), int(249 + vertical_rotation * face_length))
    right_eye = (int(292 + horizontal_rotation * face_length), int(249 + vertical_rotation * face_length))
    mouth = (int(256 + horizontal_rotation * face_length / 2), int(310 + vertical_rotation * face_length / 2))
    
    cv2.circle(img, center, 100, 0, 1)
    cv2.circle(img, left_eye, 15, 0, 1)
    cv2.circle(img, right_eye, 15, 0, 1)
    cv2.circle(img, mouth, 5, 0, 1)
    
    return img


if __name__ == '__main__':
    # 设置标准脸特征作为原点
    standard_face_image = cv2.imread('./std_face.jpg')
    if standard_face_image is None:
        print("未能加载标准脸图像。请检查路径和文件格式。")
        exit(1)
    
    standard_face_location = get_largest_face_location(standard_face_image)
    if standard_face_location is None:
        print("在标准脸图像中未检测到人脸。")
        exit(1)
    
    standard_face_keypoints = extract_facial_landmarks(standard_face_image, standard_face_location)
    if not standard_face_keypoints:
        print("未能提取标准脸图像中的人脸关键点。")
        exit(1)
    
    standard_face_features = calculate_face_features(standard_face_keypoints)

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("未能读取摄像头帧。")
            break
        """让frame镜像显示"""

        # 检测人脸并提取关键点
        face_location = get_largest_face_location(frame)
        if face_location:
            keypoints = extract_facial_landmarks(frame, face_location)
            if keypoints:
                features = calculate_face_features(keypoints)

                # 将元组转换为NumPy数组后计算相对特征量
                relative_features = features - standard_face_features

                # 绘制Vtuber脸
                vtuber_face = draw_vtuber_face(*relative_features)

                # 显示Vtuber脸
                cv2.imshow('Vtuber Face', vtuber_face)

        # 显示原始帧
        frame = cv2.flip(frame, 1)
        cv2.imshow('Original Frame', frame)

        # 按ESC键退出
        if cv2.waitKey(1) == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()