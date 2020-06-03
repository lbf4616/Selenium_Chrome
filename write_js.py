import glob
import os
import time
import numpy as np
import cv2
from openvino.inference_engine import IECore

def draw_rectangle(img, box_arr, color=(0, 0, 255), font_size=3):
    """Draw rectangles in a big image for visualization of cutting result.
    Args:
        img: An image.
        box_arr: A 2-d box array, e.g., [[x1, y1, x2, y2], [x1, y1, x2, y2], ...].
        color and font_size are optional.
    Returns:
        An image with specific bounding boxes.
    """
    if len(box_arr) == 0:
        return img
    if len(img.shape) == 2:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr_img = np.copy(img)
    for idx, box in enumerate(box_arr):
        x1, y1, x2, y2 = [int(val) for val in box]
        cv2.putText(bgr_img, str(idx), (x1, y1),  cv2.FONT_HERSHEY_SIMPLEX, 5,
                    (0, 255, 0), thickness=2,lineType=cv2.LINE_AA)
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, font_size)
    return bgr_img

def get_avg_w_h(box_list):
    """Get average width and height of all boxes in box_list.
    """
    box_arr = np.array(box_list)
    W_arr = box_arr[:, 2] - box_arr[:, 0]
    Y_arr = box_arr[:, 3] - box_arr[:, 1]
    w_mean = np.mean(W_arr)
    h_mean = np.mean(Y_arr)
    return w_mean, h_mean

def check_for_small_dead_piece(box, img, empirical_pixel_mean_thresh=70):
    """Check whether the image piece is dead piece or not.
    Args:
        box: Coordinates of an image piece, formatted as [x1, y1, x2, y2].
        img: The original image.
        empirical_pixel_mean_thresh: An empirical threshold of deadpiece, if the mean
            pixel value is less than this threshold, then the piece is dead piece.
    Returns:
        ng_flag: True for dead piece, and False otherwise.
    """
    ng_flag = False
    x1, y1, x2, y2 = box
    small_box = np.copy(img[int(y1):int(y2), int(x1):int(x2), :])
    mean_pixel_val = np.sum(small_box) / np.size(small_box)
    if mean_pixel_val < empirical_pixel_mean_thresh:
        ng_flag = True
    return ng_flag


class PieceCutter(object):
    def __init__(self, model_path, ptoduct_type, input_shape=[512, 2048, 3], piece_width=590, black_l_pixel=40):
        """
        Arg：
            model_path: 模型路径
            ptoduct_type: 产品型号
            input_shape: 模型输入大小
            gpu_id: 设置使用的gpu ID号
            gpu_mem_ratio: 设置gpu占用率
            piece_width：电池片的宽度
            black_l_pixel:电池片的左右的外边框的宽度

        """
        self.input_shape = input_shape
        self.datatype = ptoduct_type
        self.aspect_ratio_al_low = 0.05
        self.aspect_ratio_al_high = 0.05
        self.for_final_boxes_add_pixel = 15
        self.morphology_uint_size = 5
        self.morphology_uint_iter = 5
        self.ratio = 0.85
        self.empirical_pixel_mean_thresh = 70
        self.bigdeadnum = 3
        self.rough_box_height = piece_width
        self.black_l_pixel = black_l_pixel

        ie = IECore()
        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.exec_net = ie.load_network(network=net, device_name='CPU')
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))


    def predict(self, x):
        x = x.transpose((0, 3, 1, 2))
        res_net = self.exec_net.infer(inputs={self.input_blob: x})
        y = res_net[self.out_blob]
        return y

    @property
    def aspect_ratio_low(self):
        ratio = 1 - self.aspect_ratio_al_low
        return ratio

    @property
    def aspect_ratio_high(self):
        ratio = 1 + self.aspect_ratio_al_high
        return ratio

    def get_mask_from_unet(self, img):
        """Predict the mask of an image by unet."""
        img_resized = cv2.resize(
            img, (self.input_shape[1], self.input_shape[0]))
        img_test = img_resized[np.newaxis, :, :, :]
        pred_mask = self.predict(img_test)
        return pred_mask

    def extract_boxes_from_mask(self, img, pred_mask):
        """Extract boxes from the predicted mask.
        Args:
            img: Original input image.
            pred_mask: Predicted mask of the image.
        Returns:
            box_list: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """

        pred_mask_squeezed = np.squeeze(pred_mask)
        pred_mask_upsampled = cv2.resize(
            pred_mask_squeezed, (img.shape[1], img.shape[0]))
        pred_mask_binary = np.where(pred_mask_upsampled > 0.5, 1, 0)

        box_list = self.get_box_list(pred_mask_binary, al=self.for_final_boxes_add_pixel)
        return box_list

    def get_box_list(self, mask, al=15):
        """A support function of Func 'extract_boxes_from_mask', used to extract mask of each image piece.
        Args:
            mask: A binary mask image of model prediction.
            al: The number of pixels to expand.
        Returns:
            Coordinates of all connected domains.

        Note that unqualified boxes will be filtered out.
        Note that The result of cv2.contourArea() is the sum of the pixel values of all white。
        """
        box_list = []
        mask_size = np.shape(mask)
        max_h = mask_size[0] / (self.datatype[1] + 1)
        #针对组件与组件或者边缘有粘连的情况，通过设置阈值将这种情况分开，其行阈值为adhesion_width，列阈值adhesion_hight
        adhesion_width = 2.1*max_h
        adhesion_hight = max_h

        kernel = np.ones((self.morphology_uint_size, self.morphology_uint_size), np.uint8)
        mask = cv2.erode(mask.astype("uint8"), kernel, iterations=self.morphology_uint_iter)
        size = np.shape(mask)
        # 计算mask每行的像素个数，如果像素小于两个组件的宽度 就将哪一行的像素设置为0
        h_sum = mask.sum(axis=1)
        h_keep_idxs = np.where(h_sum < adhesion_width)
        for idx in h_keep_idxs:
            mask[idx, :] = 0
        # 计算mask每列的像素个数，如果像素个数小于一个组件的宽度，就将那一列的像素设置为0
        w_sum = mask.sum(axis=0)
        w_keep_idxs = np.where(w_sum < adhesion_hight)
        for idx in w_keep_idxs:
            mask[:, idx] = 0

        contour_res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_res[0]

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if float(area) < max_h*0.3 * max_h*0.3:
                cv2.drawContours(mask, [contours[i]], 0, 0, -1)

        contour_res = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_res[0]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < max_h:
                continue
            if x < 0:
                x = size[1] + x
            if y < 0:
                y = size[0] + y
            x_k = max(x - al, 0)
            y_k = max(y - al, 0)
            x_k1 = min(x + w + al, size[1])
            y_k1 = min(y + h + al, size[0])
            box_list.append([x_k, y_k, x_k1, y_k1])

        return box_list

    def add_box(self, pre_boxes, gt_boxes, mask):
        """
        Supplementary box.

        Suppose that the frame of one row of components can be detected perfectly, and the other row of components can
        not be detected completely, then the vertical coordinates of the components in the previous row will be used to
        supplement the frame, and the horizontal coordinates of the detected frame in the current row will be used to
        average the horizontal coordinates of the detected frame.
        Args:
            pre_boxes: The boxes of the component in the line to be added
            gt_boxes: The line with the full boxes
            mask:  Mainly used to obtain the size of the image
        return:
            full boxes
        NOTES： self.black_l_pixel： 为了防止大块区域死片，由于边框的宽度，造成补组件框多1个，因此使用self.black_l_pixel作为边框的经验值，去除边框宽度过大的影响。
        """
        line_u = int(np.mean([i[1] for i in pre_boxes]))
        line_d = int(np.mean([i[3] for i in pre_boxes]))
        dx_line = line_d - line_u
        dy_line = dx_line / 2
        imgsize = np.shape(mask)
        if pre_boxes[0][0] / dy_line > 1:
            space = int(round((pre_boxes[0][0]-self.black_l_pixel) / dy_line))
            if space > 0:
                for b in range(space):
                    add_box = [gt_boxes[space - b - 1][0], line_u, gt_boxes[space - b - 1][2], line_d]
                    pre_boxes.insert(0, add_box)

        if (imgsize[1] - pre_boxes[-1][2]) > dy_line:
            space = int(round((imgsize[1] - pre_boxes[-1][2]-self.black_l_pixel) / dy_line))
            if space > 0:
                for b in range(1, space + 1):
                    add_box = [gt_boxes[-b][0], line_u, gt_boxes[-b][2], line_d]
                    pre_boxes.append(add_box)

        if len(pre_boxes) == self.datatype[0]:
            return pre_boxes
        for num, _ in enumerate(pre_boxes):
            if num >= len(pre_boxes) - 1:
                break

            space = int(round((pre_boxes[num + 1][0] - pre_boxes[num][0]) / dy_line))
            space1 = (pre_boxes[num + 1][0] - pre_boxes[num][2]) / dy_line
            if space > 1 and space1 > (self.ratio - 0.05):
                con = -1
                for b in range(1, space):
                    con = con + 1
                    add_box = [gt_boxes[num + b][0], line_u, gt_boxes[num + b][2], line_d]
                    pre_boxes.insert(num + 1 + con, add_box)

        return pre_boxes

    def add_box_ner(self, pre_boxes, mask):
        """
        The number of all row boxes is less than the number of actual components of the corresponding row
        Args:
            pre_boxes: boxes to be added,eg [[x1,y1,x2,y2],...]
            mask: to get the size of the image
        return:
            Completed boxes,eg [[x1,y1,x2,y2],...]
        """
        line_u = int(np.mean([i[1] for i in pre_boxes]))
        line_d = int(np.mean([i[3] for i in pre_boxes]))
        dx_line = (line_d - line_u) / 2

        imgsize = np.shape(mask)

        if pre_boxes[0][0] / dx_line > 1:
            space = int(round((pre_boxes[0][0]-self.black_l_pixel) / dx_line))

            jz_box = pre_boxes[0][0].copy()
            if space > 0:
                for b in range(space):
                    add_box = [jz_box - (b + 1) * dx_line, line_u, jz_box - b * dx_line,
                               line_d]
                    pre_boxes.insert(0, add_box)

        if imgsize[1] - pre_boxes[-1][2] > dx_line:
            space = int(round(((imgsize[1] - pre_boxes[-1][2]) - self.black_l_pixel) / dx_line))
            if space > 0:
                jz_box = pre_boxes[-1].copy()
                for b in range(1, space + 1):
                    add_box = [jz_box[0] + b * dx_line, line_u, jz_box[2] + b * dx_line,
                               line_d]
                    pre_boxes.append(add_box)
        if len(pre_boxes) == self.datatype[0]:
            return pre_boxes

        for num, box in enumerate(pre_boxes):
            if num >= len(pre_boxes) - 1:
                break
            space = int(round((pre_boxes[num + 1][0] - pre_boxes[num][0]) / dx_line))

            space1 = (pre_boxes[num][2] - pre_boxes[num][0]) / dx_line
            if space > 1 and space1 > (self.ratio - 0.05):
                ystart = box[2]
                con = -1
                for b in range(space - 1):
                    con = con + 1
                    add_box = [ystart + dx_line * (b), line_u, ystart + (dx_line * (b + 1)), line_d]
                    pre_boxes.insert(num + 1 + con, add_box)

        return pre_boxes

    def check_box_num(self, box_list, img):
        """
        Check the number of bbox.
        Args:
            box_list: [[x1,y1,x2,y2],[],]
            mask: Mainly to obtain image size
        return:
            right_list:Correct image bbox
        """
        box_list = np.array(box_list)
        gt_box_num = self.datatype[0] * self.datatype[1]

        if len(box_list) == gt_box_num:
            return box_list
        elif len(box_list) > gt_box_num:
            return [[0, 0, 0, 0]]
        elif len(box_list) < gt_box_num:
            first_line = {}
            secord_line = {}
            f_line = []
            s_line = []
            size = np.shape(img)
            dy_line = size[0] / np.floor(self.datatype[1]+1)
            for box in box_list:
                if box[1] < dy_line:
                    first_line[box[0]] = box
                else:
                    secord_line[box[0]] = box
            sorted(secord_line.items(), key=lambda item: item[0], reverse=False)
            for _, v in sorted(first_line.items(), key=lambda item: item[0], reverse=False):
                f_line.append(v)
            for _, v in sorted(secord_line.items(), key=lambda item: item[0], reverse=False):
                s_line.append(v)

            if len(f_line) == 0 or len(s_line) == 0:
                return [[0, 0, 0, 0]]

            if len(f_line) < self.datatype[0] and len(s_line) == self.datatype[0]:
                right_f = self.add_box(f_line, s_line, img)
                right_list = list(right_f) + list(s_line)
            elif len(f_line) == self.datatype[0] and len(s_line) < self.datatype[0]:
                right_s = self.add_box(s_line, f_line, img)
                right_list = list(right_s) + list(f_line)
            else:
                if len(f_line) > len(s_line):
                    right_f = list(self.add_box_ner(f_line, img))
                    right_s = list(self.add_box(s_line, right_f, img))
                else:
                    right_f = list(self.add_box_ner(s_line, img))
                    right_s = list(self.add_box(f_line, right_f, img))
                right_list = right_f + right_s

            return right_list

    def divide_connected_box(self, box, rough_box_height):
        """Divide the box if it consists of several regular boxes.
        Args:
            box: The input box.
            rough_box_height: Calculated by dividing the height of the original image by 6.
        Returns:
            result_box: Return a 2-d list, which consists of several regular boxes. If the
                input box can not be divided, return [box].
        """
        result_box = []
        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1
        assert (x1 <= x2 and y1 <= y2)
        rough_cols = int(round(w_box / (rough_box_height / 2)))
        rough_rows = int(round(h_box / rough_box_height))

        if rough_cols == 0 or rough_rows == 0:
            result_box.append([0, 0, 0, 0])
        elif rough_cols == 1 and rough_rows == 1:
            result_box.append(box)
        elif max(w_box, h_box) / min(w_box, h_box) < self.aspect_ratio_low:
            result_box.append(box)
        else:
            avg_w = (x2 - x1) / rough_cols
            avg_h = (y2 - y1) / rough_rows
            for col_ind in range(rough_cols):
                for row_ind in range(rough_rows):
                    tmp_x1 = x1 + col_ind * avg_w
                    tmp_y1 = y1 + row_ind * avg_h
                    tmp_x2 = x1 + (col_ind + 1) * avg_w
                    tmp_y2 = y1 + (row_ind + 1) * avg_h
                    tmp_box = [tmp_x1, tmp_y1, tmp_x2, tmp_y2]
                    result_box.append(tmp_box)
        return result_box

    def rectify_mask_box_list(self, ori_box_list, rough_box_height, al=0):
        """Rectify box list extracted from mask.
        Args:
            ori_box_list: Box list extracted from mask.
            rough_box_height: Calculated by dividing the height of the original image by np.floor(self.datatype[1]).
        Returns:
            box_list: Rectified box list.
        """
        box_list = []
        for ori_box in ori_box_list:
            divided_boxes = self.divide_connected_box(
                ori_box, rough_box_height)
            for box in divided_boxes:
                box_normed = [int(box[0] - al), int(box[1] - al),
                              int(box[2] + al), int(box[3] + al)]
                box_list.append(box_normed)
        return box_list

    def check_for_dead_piece_big(self, img, box_list, row_num=6):
        """Check whether the big image is dead or not.

        Note that in case 4, we divide the big image into 6*12 pieces, no matter the original
        big image is 6*12 or 6*10. We only detect possible dead pieces herein.
        Args:
            img: The original image.
            box_list: Rectified cutted box list
        Returns:
            ng_flag: True for dead piece, and False otherwise.
        """
        box_arr = np.array(box_list)
        min_x1, min_y1, _, _ = np.min(box_arr, 0)
        _, _, max_x2, max_y2 = np.max(box_arr, 0)
        boundary_limit = img.shape[0] / (row_num + 1)
        if (img.shape[0] - max_y2) > boundary_limit or (img.shape[1] - max_x2) > boundary_limit or (
                min_x1 - 0) > boundary_limit or (min_y1 - 0) > boundary_limit:
            ng_flag = True
            return ng_flag

    def check_for_dead_piece_small(self, img, box_list, dead_piece_abnormal, empirical_pixel_mean_thresh=70):
        ng_flag = False
        for num, box in enumerate(box_list):
            if check_for_small_dead_piece(box, img, empirical_pixel_mean_thresh=empirical_pixel_mean_thresh):
                dead_piece_abnormal[num] = 1
                ng_flag = True
        return ng_flag, dead_piece_abnormal

    def image_segment(self, img):
        """
        切图的主函数
         Args:
            img:待切割的图片
        return:
            cut_ok_flag:img是否是OK图片
            dead_piece_abnormal:返回一个onehot向量（电池片列数×电池片行数），比如电池片为[24,2],其形式输出为[0,0,0,0,0,...]
            piece_disorder_abnormal:电池片拼接异常或组件叠加在一起时，返回True，反之，正常返回False
            cut_box_list:每个组件的bbox。在图片切割正常时，其形式[[x1,y1,x2,y2],[x1,y1,x2,y2]...]；在图片切割异常时，其形式[]
            h_mean：每个组件的高
            w_mean：每个组件的宽

        基本逻辑：
        1、只要cut_ok_flag = True，这张图片一定可以切割且没有死片，同时dead_piece_abnormal为全零且cut_box_list返回所有组件的框。
        2、当存在死片或组件叠加时，cut_ok_flag = False，这是会存在两种情况
        （1）零散的小死片：dead_piece_abnormal向量不全为零，有死片的位置为1，另外cut_box_list返回所有组件和死片的框
        （2）大死片和组件重叠：dead_piece_abnormal向量全为1，另外cut_box_list返回[]
        """
        cut_ok_flag = True
        dead_piece_abnormal = np.zeros(self.datatype[0] * self.datatype[1])
        piece_disorder_abnormal = False
        h_mean = self.rough_box_height
        w_mean = h_mean / 2
        pred_mask = self.get_mask_from_unet(img)

        rough_box_height = self.rough_box_height
        ori_box_list = self.extract_boxes_from_mask(img, pred_mask)
        if len(ori_box_list) < self.bigdeadnum:
            cut_ok_flag = False
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        cut_box_list = self.rectify_mask_box_list(ori_box_list, rough_box_height)
        if [0, 0, 0, 0] in cut_box_list:
            cut_ok_flag = False
            piece_disorder_abnormal = True
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        box_list = self.check_box_num(cut_box_list, img)
        if len(box_list) < self.bigdeadnum:
            cut_ok_flag = False
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        w_mean, h_mean = get_avg_w_h(box_list)
        w_mean_safe = w_mean - w_mean / 20
        h_mean_safe = h_mean - h_mean / 20
        box_list = sorted(box_list, key=(
            lambda x: [int(x[1] / h_mean_safe), int(x[0] / w_mean_safe)]), reverse=False)
        if self.check_for_dead_piece_big(img, box_list, row_num=np.floor(self.datatype[1])):
            cut_ok_flag = False
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        if len(box_list) > self.datatype[0] * self.datatype[1]:
            cut_ok_flag = False
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean

        if len(box_list) < self.datatype[0] * self.datatype[1]:
            cut_ok_flag = False
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        ng_flag, dead_piece_abnormal = self.check_for_dead_piece_small(img, box_list, dead_piece_abnormal,
                                                                     empirical_pixel_mean_thresh=self.empirical_pixel_mean_thresh)
        if np.sum(dead_piece_abnormal) > 0.25 * self.datatype[0] * self.datatype[1]:
            cut_ok_flag = False
            dead_piece_abnormal = np.ones(self.datatype[0] * self.datatype[1])
            cut_box_list = []
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        if ng_flag:
            cut_ok_flag = False
            dead_piece_abnormal = dead_piece_abnormal
            cut_box_list = box_list
            return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean
        else:
            cut_box_list = box_list
        return cut_ok_flag, dead_piece_abnormal, piece_disorder_abnormal, cut_box_list, h_mean, w_mean


if __name__ == '__main__':
    from tqdm import tqdm
    import pandas as pd
    from base64 import b64encode
    from json import dumps
    import json

    ptoduct_type1=[20, 2]
    small_Ct = PieceCutter(model_path='./model/unet_for_cutting.xml', ptoduct_type = ptoduct_type1)
    print("--****--")
    output = './output/'
    data_root = '/home/blin/Pictures/数据拷贝0601/20/'
    csv_data = pd.read_csv("clean_data_120.csv")

    last_name = str(csv_data.iloc[0][8])
    for i in tqdm(range(0, csv_data.shape[0])):
        print(i, csv_data.iloc[i][8])
        name = str(csv_data.iloc[i][8])
        im_path = data_root + str(csv_data.iloc[i][8]) + '.jpg'
        im = cv2.imread(im_path)
        try:
            im1 = im[:1259]
            im2 = im[1259:2460]
            im3 = im[2460:]
            cut_ok_flag1, dead_piece_abnormal1, piece_disorder_abnormal1, cut_box_list1, h_mean1, w_mean1 = small_Ct.image_segment(im1)
            cut_ok_flag2, dead_piece_abnormal2, piece_disorder_abnormal2, cut_box_list2, h_mean2, w_mean2 = small_Ct.image_segment(im2)
            cut_ok_flag3, dead_piece_abnormal3, piece_disorder_abnormal3, cut_box_list3, h_mean3, w_mean3 = small_Ct.image_segment(im3)
            # print(len(cut_box_list1), len(cut_box_list2), len(cut_box_list3))
            tt_len = len(cut_box_list1) + len(cut_box_list2) + len(cut_box_list3)
            if tt_len != ptoduct_type1[0] * 6:
                continue
        except:
            continue
        bad_reason = csv_data.iloc[i][0]
        bad_locations = csv_data.iloc[i][1]
        print(bad_reason)
        # print(bad_locations)

        bad_location_list = bad_locations.split(",")
        bad_location_list = sorted(bad_location_list)
        print(bad_location_list)

        data = {
                "version": "4.2.10",
                "flags": {},
                "shapes": [],
                "imagePath": "1.jpg",
                "imageData": "",
                }
        im_name = name + '.jpg'
        with open(im_path, 'rb') as img_file:
            byte_content = img_file.read()

        base64_bytes = b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
        data["imageData"] = base64_string
        data["imagePath"] = im_name
        
        if i > 0 and last_name == name:
            f = open(data_root + str(last_name) + '.json', encoding='utf-8')
            data = json.load(f)

        last_name = name
        for bad_location in bad_location_list:
            if bad_location[0] == 'A' or bad_location[0] == 'B':
                if bad_location[0] == 'A':
                    real_idx = 0
                else:
                    real_idx = ptoduct_type1[0]

                bad_points = cut_box_list1[int(bad_location[1:])-1 + real_idx]

                shapes = {
                    "label": "",
                    "points": [
                        [
                        393.7120418848167,
                        221.9895287958115
                        ],
                        [
                        432.4554973821989,
                        254.45026178010468
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                    }
                shapes["label"] = bad_reason
                shapes["points"] = [[int(bad_points[0]), int(bad_points[1])], [int(bad_points[2]), int(bad_points[3])]]
                data["shapes"].append(shapes)

            if bad_location[0] == 'C' or bad_location[0] == 'D':
                if bad_location[0] == 'C':
                    real_idx = 0
                else:
                    real_idx = ptoduct_type1[0]

                bad_points = cut_box_list2[int(bad_location[1:])-1 + real_idx]
                shapes = {
                    "label": "",
                    "points": [
                        [
                        393.7120418848167,
                        221.9895287958115
                        ],
                        [
                        432.4554973821989,
                        254.45026178010468
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                    }

                bad_points[1] = bad_points[1] + 1259
                bad_points[3] = bad_points[3] + 1259
                shapes["label"] = bad_reason
                shapes["points"] = [[int(bad_points[0]), int(bad_points[1])], [int(bad_points[2]), int(bad_points[3])]]
                data["shapes"].append(shapes)

            if bad_location[0] == 'E' or bad_location[0] == 'F':
                if bad_location[0] == 'E':
                    real_idx = 0
                else:
                    real_idx = ptoduct_type1[0]

                bad_points = cut_box_list3[int(bad_location[1:])-1 + real_idx]
                shapes = {
                    "label": "",
                    "points": [
                        [
                        393.7120418848167,
                        221.9895287958115
                        ],
                        [
                        432.4554973821989,
                        254.45026178010468
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                    }
                bad_points[1] = bad_points[1] + 2460
                bad_points[3] = bad_points[3] + 2460
                shapes["label"] = bad_reason
                shapes["points"] = [[int(bad_points[0]), int(bad_points[1])], [int(bad_points[2]), int(bad_points[3])]]
                data["shapes"].append(shapes)

        write_data = dumps(data)
        
        js = im_path.replace('.jpg', '.json')
        with open(js, 'w') as json_file:
            json_file.write(write_data)


        # for bad_location in bad_location_list:
        #     if bad_location[0] == 'A' or bad_location[0] == 'B':
        #         print(bad_location)
        #         idx = int((int(bad_location[1]) - 1) / 4) * 4
        #         print(idx)
        #         x1 = cut_box_list1[idx][0]
        #         y1 = cut_box_list1[idx][1]
        #         print(idx + ptoduct_type1[0] + 1)
        #         x2 = cut_box_list1[idx + ptoduct_type1[0] + 3][2]
        #         y2 = cut_box_list1[idx + ptoduct_type1[0] + 3][3]

        #         print(y1,y2, x1,x2)
        #         small_pic = im1[y1:y2, x1:x2]
        #         save_name = output + name + bad_location + '.jpg'
        #         cv2.imwrite(save_name, small_pic)
        #         ori_cor = cut_box_list1[int(bad_location[1])-1]
        #         first_cor = cut_box_list1[idx]
        #         x1 = min(ori_cor[0], ori_cor[2]) - min(first_cor[0], first_cor[2])
        #         x2 = max(ori_cor[0], ori_cor[2]) - min(first_cor[0], first_cor[2])
        #         y1 = 0
        #         y2 = max(ori_cor[1],ori_cor[3]) - min(ori_cor[1],ori_cor[3])
        #         print(x1,y1, x2, y2)
        #         small_cor = [[int(x1), int(y1)],[int(x2), int(y2)]]

        # img_with_bbox = draw_rectangle(im1, cut_box_list1, color=(0, 0, 255), font_size=3)

        # cv2.imwrite("11.jpg",img_with_bbox)