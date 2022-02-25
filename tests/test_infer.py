import sys
sys.path.append('./')
sys.path.append('./nuclei')
from segm.infer import process as segmenter_infer
from nuclei.nuclei_infer import process as nuclei_infer
from segm.data.slide import read_mat_file
import cv2
import matplotlib.pyplot as plt


def test_segmenter_infer():
    model_path = '/mnt/d/segmenter/runs/seg_tiny_unet_1/checkpoint.pth'
    input_dir = '/mnt/d/segmenter/dataset/enzo_test_1/images'
    output_dir = '/mnt/d/segmenter/segmaps'
    segmenter_infer(model_path, input_dir, output_dir)


def test_nuclei_infer():
    data_folder = 'segm/data/enzo_test/images'
    model_name = 'nucles_model_v3.meta'
    format = '.png'
    nuclei_infer(data_folder, model_name, format)


def get_label_gt():
    image_file = 'segm/data/enzo_test/images/consep_1.png'
    label_file = 'segm/data/enzo_test/annotations/consep_1.mat'
    preidct_path = 'segm/segmaps/_mul_mask.png'
    # Read image
    img = cv2.imread(image_file)
    label = read_mat_file(label_file, 500)

    preidct_mask = cv2.imread(preidct_path, 0)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(label)
    fig.add_subplot(1, 2, 2)
    plt.imshow(preidct_mask)
    plt.show()


if __name__ == "__main__":
    test_segmenter_infer()
    # show image
    # image = cv2.imread('/mnt/d/segmenter/segmaps/consep_1.png', 0)
    # plt.imshow(image)
    # plt.show()

    # test_nuclei_infer()
    # get_label_gt()