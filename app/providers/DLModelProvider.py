import cv2
import numpy as np
import tensorflow as tf
from masonite.provider import ServiceProvider

from config import storage

index_to_char = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
    "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", ""
]


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def locate(img_src, img_mask):
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取边缘区域宽度最大的区域
    cont = max(contours, key=lambda item: cv2.boundingRect(item)[2])

    x, y, w, h = cv2.boundingRect(cont)  # 获取最小外接矩形

    if w > 15 and h > 15:
        rect = cv2.minAreaRect(cont)  # 针对坐标点获取带方向角的最小外接矩形，中心点坐标，宽高，旋转角度
        box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标

        cont = cont.reshape(-1, 2).tolist()
        # 由于转换矩阵的两组坐标位置需要一一对应，因此需要将最小外接矩形的坐标进行排序，最终排序为[左上，左下，右上，右下]
        box = sorted(box, key=lambda xy: xy[0])  # 先按照左右进行排序，分为左侧的坐标和右侧的坐标
        box_left, box_right = box[:2], box[2:]  # 此时box的前2个是左侧的坐标，后2个是右侧的坐标
        box_left = sorted(box_left, key=lambda x: x[1])  # 再按照上下即y进行排序，此时box_left中为左上和左下两个端点坐标
        box_right = sorted(box_right, key=lambda x: x[1])  # 此时box_right中为右上和右下两个端点坐标
        box = np.array(box_left + box_right)  # [左上，左下，右上，右下]

        x0, y0 = box[0][0], box[0][1]  # 这里的4个坐标即为最小外接矩形的四个坐标，接下来需获取平行(或不规则)四边形的坐标
        x1, y1 = box[1][0], box[1][1]
        x2, y2 = box[2][0], box[2][1]
        x3, y3 = box[3][0], box[3][1]

        def point_to_line_distance(X, Y):
            if x2 - x0:
                k_up = (y2 - y0) / (x2 - x0)  # 斜率不为无穷大
                d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
            else:  # 斜率无穷大
                d_up = abs(X - x2)
            if x1 - x3:
                k_down = (y1 - y3) / (x1 - x3)  # 斜率不为无穷大
                d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
            else:  # 斜率无穷大
                d_down = abs(X - x1)
            return d_up, d_down

        d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
        l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        for each in cont:  # 计算cont中的坐标与矩形四个坐标的距离以及到上下两条直线的距离，对距离和进行权重的添加，成功选出四边形的4个顶点坐标
            x, y = each[0], each[1]
            dis0 = (x - x0) ** 2 + (y - y0) ** 2
            dis1 = (x - x1) ** 2 + (y - y1) ** 2
            dis2 = (x - x2) ** 2 + (y - y2) ** 2
            dis3 = (x - x3) ** 2 + (y - y3) ** 2
            d_up, d_down = point_to_line_distance(x, y)
            weight = 0.975
            if weight * d_up + (1 - weight) * dis0 < d0:
                d0 = weight * d_up + (1 - weight) * dis0
                l0 = (x - 4, y - 2)
            if weight * d_down + (1 - weight) * dis1 < d1:
                d1 = weight * d_down + (1 - weight) * dis1
                l1 = (x - 4, y + 2)
            if weight * d_up + (1 - weight) * dis2 < d2:
                d2 = weight * d_up + (1 - weight) * dis2
                l2 = (x + 4, y - 2)
            if weight * d_down + (1 - weight) * dis3 < d3:
                d3 = weight * d_down + (1 - weight) * dis3
                l3 = (x + 4, y + 2)

        p0 = np.float32([l0, l1, l2, l3])  # 左上角，左下角，右上角，右下角，形成的新box顺序需和原box中的顺序对应，以进行转换矩阵的形成
        p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])
        transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
        lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # 进行车牌矫正

        return lic


class DLModelProvider(ServiceProvider):
    wsgi = True
    __detect_model = None
    __recognition_model = None
    __storage_location = ''

    def register(self):
        self.app.bind('ai', self)
        self.__detect_model = tf.keras.models.load_model('zc.h5')
        self.__recognition_model = tf.keras.models.load_model('plate.h5')
        self.__storage_location = storage.DRIVERS['disk']['location']

    def predict(self, filename):
        img = tf.io.read_file(self.__storage_location + '/' + filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [416, 416])
        img = img / 255.0

        result = self.__detect_model.predict(np.array([img]))

        mask = create_mask(result)
        mask = tf.keras.preprocessing.image.array_to_img(mask)
        mask = np.asarray(mask)
        img = np.asarray(img)

        plate_image = locate(img, mask)
        plate_chars = self.__recognition_model.predict(np.array([plate_image]))
        plate = []
        for cs in plate_chars:
            plate.append(index_to_char[np.argmax(cs)])

        return ''.join(plate)
