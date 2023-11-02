# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import cv2
import numpy as np

class Display(object):
    def __init__(self, width, height, bg_color=(255, 255, 255)):
        self._buf = np.ones((height, width, 3), dtype=np.uint8)
        self.bg_color = bg_color
        self._buf[:, :] = bg_color

    def place_rgb(self, y, x, image):
        self._buf[x:x+image.shape[1], y:y+image.shape[0], :] = image[:]

    def place_rgb_float(self, y, x, image):
        self._buf[x:x + image.shape[1], y:y + image.shape[0], :] = (255*image[:]).astype(np.uint8)


    def place_gray(self, y, x, image):
        self._buf[x:x + image.shape[1], y:y + image.shape[0], 0] = image[:]
        self._buf[x:x + image.shape[1], y:y + image.shape[0], 1] = image[:]
        self._buf[x:x + image.shape[1], y:y + image.shape[0], 2] = image[:]

    def place_gray_float(self, y, x, image):
        self._buf[x:x + image.shape[1], y:y + image.shape[0], 0] = (255 * image[:]).astype(np.uint8)
        self._buf[x:x + image.shape[1], y:y + image.shape[0], 1] = (255 * image[:]).astype(np.uint8)
        self._buf[x:x + image.shape[1], y:y + image.shape[0], 2] = (255 * image[:]).astype(np.uint8)

    def show(self, winname="Window"):
        cv2.imshow(winname=winname,  mat=self._buf)

    def write(self, filename):
        cv2.imwrite(filename, self._buf)


class FancyDisplay(object):
    def __init__(self, padding=30, bg_color=(255, 255, 255)):
        self.pictures = []
        self.buf = None
        self.padding=padding
        self.bg_color = bg_color

    def add_picture(self, name, height, width, row, column):
        """
        If column is -1 then the picture will be automatically placed in the next
        available column
        :param name:
        :param height:
        :param width:
        :param row:
        :param column:
        :return:
        """
        self.pictures.append((name, height, width, row, column))

    def initialize(self):
        xdim = 0
        ydim = 0
        rows = 0
        columns = 0
        pic1=[]
        for (i, (name, height, width, row, column)) in enumerate(self.pictures):
            if column == -1:
                last_c = -1
                for (n, h, w, r, c) in self.pictures[:i]:
                    if r == row and c>last_c:
                        last_c = c
                column = last_c + 1
            pic1.append((name, height, width, row, column))
            self.pictures = pic1 + self.pictures[i:]
        self.pictures = pic1

        for (name, height, width, row, column) in self.pictures:
            if row > rows:
                rows = row
            if column > columns:
                columns = column
        rows += 1
        columns += 1
        y_rows = [self.padding]
        x_cols = []
        for r in range(rows):
            x_column = [0]
            w_column = [0]
            h_row = []
            for c in range(columns):
                for (name, height, width, row, column) in self.pictures:
                    if row == r and column == c:
                        x_column.append(x_column[-1]+self.padding+w_column[-1])
                        w_column.append(width)
                        h_row.append(height)
                        if x_column[-1]+width+self.padding>xdim:
                            xdim=x_column[-1]+width+self.padding
            y_rows.append(y_rows[-1]+max(h_row)+self.padding)
            x_cols.append((x_column[1:]))
            if y_rows[-1]>ydim:
                ydim = y_rows[-1]
        self.y_rows = y_rows
        self.x_cols = x_cols
        self._buf = np.ones((ydim, xdim, 3), dtype=np.uint8)
        self._buf[:, :] = self.bg_color
        self.pic_hash = {}
        self.pic_hash_hw = {}
        for (name, height, width, row, column) in self.pictures:
            self.pic_hash[name] = [self.x_cols[row][column], self.y_rows[row]]
            self.pic_hash_hw[name] = [height, width]
            py = self.pic_hash[name][0]
            px = self.pic_hash[name][1]
            cv2.rectangle(self._buf, (py-2, px-2), (py+width+1, px+height+1), color=(140, 100, 100))
            cv2.putText(self._buf, name, (py-1, px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), lineType=cv2.LINE_AA)

    def place_picture(self, name, image, flip_bgr=False):
        if image.dtype == np.float or image.dtype == np.float32:
            if len(image.shape) == 3:
                self.place_rgb_float(self.pic_hash[name][0], self.pic_hash[name][1], image, flip_bgr=flip_bgr)
            else:
                self.place_gray_float(self.pic_hash[name][0], self.pic_hash[name][1], image)
        else:
            if len(image.shape) == 3:
                self.place_rgb(self.pic_hash[name][0], self.pic_hash[name][1], image, flip_bgr=flip_bgr)
            else:
                self.place_gray(self.pic_hash[name][0], self.pic_hash[name][1], image)

    def place_rgb(self, y, x, image, flip_bgr=False):
        if flip_bgr:
            self._buf[x:x+image.shape[0], y:y+image.shape[1], :] = image[:, :, ::-1]
        else:
            self._buf[x:x+image.shape[0], y:y+image.shape[1], :] = image[:]


    def place_rgb_float(self, y, x, image, flip_bgr=False):
        if flip_bgr:
            self._buf[x:x + image.shape[0], y:y + image.shape[1], :] = (255*image[:, :, ::-1]).astype(np.uint8)
        else:
            self._buf[x:x + image.shape[0], y:y + image.shape[1], :] = (255*image[:]).astype(np.uint8)


    def place_gray(self, y, x, image):
        self._buf[x:x + image.shape[0], y:y + image.shape[1], 0] = image[:]
        self._buf[x:x + image.shape[0], y:y + image.shape[1], 1] = image[:]
        self._buf[x:x + image.shape[0], y:y + image.shape[1], 2] = image[:]

    def place_gray_float(self, y, x, image):
        self._buf[x:x + image.shape[0], y:y + image.shape[1], 0] = (255 * image[:]).astype(np.uint8)
        self._buf[x:x + image.shape[0], y:y + image.shape[1], 1] = (255 * image[:]).astype(np.uint8)
        self._buf[x:x + image.shape[0], y:y + image.shape[1], 2] = (255 * image[:]).astype(np.uint8)

    def place_txt(self, name, texts):
        image = np.ones((self.pic_hash_hw[name][0], self.pic_hash_hw[name][1], 3), dtype=np.uint8)
        image[:, :] = self.bg_color
        y = 2
        x = 10
        for text in texts:
            cv2.putText(image, text, (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), lineType=cv2.LINE_AA)
            x += 12
        self.place_rgb(self.pic_hash[name][0], self.pic_hash[name][1], image)

    def show(self, winname="Window"):
        cv2.imshow(winname=winname,  mat=self._buf)

    def write(self, filename):
        cv2.imwrite(filename, self._buf)

    def get(self):
        return self._buf


if __name__ == "__main__":
    F= FancyDisplay()
    F.add_picture("pic1", 130, 120, 0, 0)
    F.add_picture("pic2", 100, 200, 0, 1)
    F.add_picture("pic3", 50, 50, 0, 2)
    F.add_picture("pic4", 25, 25, 0, 3)
    F.add_picture("pic5", 40, 40, 1, 0)
    F.add_picture("pic5", 40, 40, 1, 1)
    F.add_picture("pic5", 40, 40, 1, 2)
    F.initialize()
    print(F.y_rows)
    print(F.x_cols)
    print(F.pic_hash)
    print(F._buf.shape)
    F.place_picture("pic1", np.zeros((130, 120), dtype=np.uint8))
    F.show("Win")
    cv2.waitKey(0)