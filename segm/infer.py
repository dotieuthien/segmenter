import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import segm.utils.torch as ptu

from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.slide import SLIDE_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb

from segm.model.factory import load_model
from segm.model.utils import inference


def load_img(img_path, img_size):
    """Load and convert image into tensor batch for inference

    Args:
        img_path ([str]): path to tile image
        img_size ([int]): image size to feedforward

    Returns:
        pil_im ([pil])
        im ([tensor])
    """
    transform = transforms.ToTensor()
    pil_im = Image.open(img_path)
    im = np.array(pil_im, dtype=np.uint8)
    im = cv2.resize(im, (img_size, img_size))
    pil_im = Image.fromarray(im)
    im = transform(im)
    return pil_im, im


# @click.command()
# @click.option("--model-path", type=str)
# @click.option("--input-dir", "-i", type=str, help="folder with input images")
# @click.option("--output-dir", "-o", type=str, help="folder with output images")
# @click.option("--gpu/--cpu", default=True, is_flag=True)
def process(model_path, input_dir, output_dir, gpu=True):
    """Infer Segmenter model

    Args:
        model_path ([str]): [description]
        input_dir ([str]): [description]
        output_dir ([str]): [description]
        gpu ([boolean]): [description]
    Return:
    """
    ptu.set_gpu_mode(gpu)

    # Model
    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    model.to(ptu.device)

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    cat_names, cat_colors = dataset_cat_description(SLIDE_CATS_PATH)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    list_dir = list(input_dir.iterdir())

    for filename in tqdm(list_dir, ncols=80):
        # pil_im = Image.open(filename).copy()
        # im = F.pil_to_tensor(pil_im).float() / 255
        # im = F.normalize(im, normalization["mean"], normalization["std"])
        pil_im, im = load_img(filename, variant["inference_kwargs"]["im_size"])
        im = im.to(ptu.device).unsqueeze(0)

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=im.shape[2:4],
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=variant["inference_kwargs"]["window_stride"],
            batch_size=1,
        )
        seg_map = logits.argmax(0, keepdim=True)
        seg = seg_map.cpu().detach().numpy().astype(np.uint8)[0]
        pil_seg = Image.fromarray(seg)
        plt.imshow(pil_seg)
        plt.show()
        pil_seg.save(output_dir / filename.name)
 
        # seg_rgb = seg_to_rgb(seg_map, cat_colors)
        # seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        # pil_seg = Image.fromarray(seg_rgb[0])

        # pil_seg.save(output_dir / filename.name)
        # pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
        # pil_blend.save(output_dir / filename.name)
    return
