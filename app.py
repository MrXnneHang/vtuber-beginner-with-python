import cv2
from utils.cal_facepoint_position_util import get_largest_face_location, extract_facial_landmarks, calculate_face_features,draw_vtuber_face

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


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