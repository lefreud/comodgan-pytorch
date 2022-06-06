from pycocotools.coco import COCO
import random
import skimage.io as io
from skimage.morphology import footprints, dilation


def get_image_pairs(annotations_file_path, *, category_names=None, image_path="datasets/coco/val2017", erosion_radius=0):
    coco_helper = COCO(annotations_file_path)
    image_ids = coco_helper.getImgIds()
    random.shuffle(image_ids)

    if category_names is None:
        category_names = []
    category_ids = coco_helper.getCatIds(catNms=category_names)

    erosion_footprint = footprints.disk(erosion_radius)

    for image_id in image_ids:
        image_info = coco_helper.loadImgs(image_id)
        annotations_ids = coco_helper.getAnnIds(imgIds=[image_id], catIds=category_ids)
        annotations = coco_helper.loadAnns(annotations_ids)

        image = io.imread(f"{image_path}/{image_info[0]['file_name']}")
        mask = coco_helper.annToMask(annotations[0])
        if erosion_radius > 0:
            mask = dilation(mask, erosion_footprint)
        yield image, mask
