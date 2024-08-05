import cv2

def capture_standard_face():
    """
    启动相机并在按下 Enter 键时拍照，保存到 ./std_face.jpg 文件中。
    """
    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按下 Enter 键以拍照，按下 Esc 键退出。")

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 显示当前帧
        cv2.imshow('Camera', frame)

        # 等待键盘事件
        key = cv2.waitKey(1)
        if key == 27:  # 按下 Esc 键退出
            break
        elif key == 13:  # 按下 Enter 键拍照
            # 保存照片
            cv2.imwrite('./std_face.jpg', frame)
            print("照片已保存到 ./std_face.jpg")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 调用函数启动相机并拍照
capture_standard_face()
