import cv2
import sys

if __name__ == '__main__' :
    tracker = cv2.TrackerMIL_create()
    # Đọc video , hàm VideoCapture là lấy video từ file và gán nó vào một biến
    video = cv2.VideoCapture("videos/hihi.mp4")
    # Kiểm tra xem video có mở được không , nếu không mở được thì thoát ra
    if not video.isOpened():
        print("Không tìm thấy video")
        sys.exit()
    # đọc frame đầu tiên của video
    ok, frame = video.read()
    if not ok:
        print("Không thể đọc video")
        sys.exit()
    # Xác định môt hộp giới hạn ban đầu , cái này mục đích như tạo một biến bau đầu thôi , ở dưới gán lại
    bbox = (287, 23, 86, 320)

    # hàm selectROI để chọn một vùng hình chữ nhật mà mình quan tâm trong hình ảnh
    # khi mở video lên , mình sẽ chọn(kéo thả) một vùng hình ảnh mà mà mình quan tâm nó theo dõi sau này
    # tham số False phía sau chỉ rõ là đoạn hình chữ nhật được lấy từ trên trái xuống dưới phải chứ ko tính từ tâm

    bbox = cv2.selectROI(frame, False)

    # Khởi tạo đầu cho tracker , tham số truyền vào là frame đầu tiên của video và hình bao mà mình muốn tracking
    # hàm trả về True hoặc False
    ok = tracker.init(frame, bbox)

    ## liên tục đọc cái khung mới từ video và cập nhật cho video
    while True:
        # Đọc một frame mới từ video , sau đó kiểm tra xem có thành công không , nếu không thì kết thúc
        ok, frame = video.read()
        if not ok:
            break

        # Start timer , lấy só xung nhịp hiện tại , từ thời điển mở máy đến thời điểm này
        timer = cv2.getTickCount()

        # Cập nhật khung mới cho Tracker theo dõi
        # tracker sẽ trả về True nếu tracking thành công , và trả về khung của hình chữ nhật đã tracking được
        ok, bbox = tracker.update(frame)
        print(type(bbox))

        # Tính toán Frames per second (FPS)
        # getTickFrequency là lấy tần số xung nhịp , hay là số xung nhịp mỗi giây
        # fps (Frames per second) ở dưới tính xem mỗi dây có bao nhiêu frame
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Kiểm tra xem kết quả traking ở khung hiện tại có thành công hay không
        # nếu thành công thì vẽ hình bao xung quanh vùng tracking được , nó ở trong bbox
        if ok:
            # bbox chứa góc trên trái và chiều dài chiều rộng của hình chữ nhật
            # cần chuyển đổi nó thành hai điểm trên trái và dưới phải để vẽ hình chữ nhật bao
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Nếu tracking thất bại thì in ra dòng chữ thất bại trong video
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)

        # hiển thị kiểu tracker
        cv2.putText(frame, "MIL_" + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Hiển thị FPS (số khung hình trong mỗi dây)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # hiển thị  kết quả ra màn hình
        cv2.imshow("Tracking", frame)

        # nếu muốn thoát thì ấn ESC
        # waitKey() luôn luôn theo sau imshow , nếu không có waitKey thì hình ảnh không được hiển thị
        # waitKey(1) tức là sẽ hiển thị hình ảnh sau mỗi 1ms
        # hàm trả về số int tương ứng với nút mình ấn , có hàm ord() để chuyển các phím  sang int , ví dụ bên dưới

        k = cv2.waitKey(1) & 0xff
        print(k)
        if k == ord("m"): break  ## k == 27 tức là sự kiện nhấn nút ESC