import os
import numpy as np
import cv2



def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)


def prepare_gray_version():
    location = "data/ham10000_images"
    for name in os.listdir(location):
        path = location + '/' + name
        _img = cv2.imread(path, cv2.IMREAD_COLOR)
        #img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        img_cc = shade_of_gray_cc(_img, 6, 2.2)
        filename = "data/gray/"+ name
        print(filename)
        cv2.imwrite(filename, img_cc)
        _img = cv2.imread(filename, cv2.IMREAD_COLOR)
        import matplotlib.pyplot as plt
        plt.imshow(_img)
        plt.show



if __name__ == "__main__":
    prepare_gray_version()
