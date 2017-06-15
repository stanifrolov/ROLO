import cv2
import os
import sys
from utils import ROLO_utils as utils

def main(argv):
  num_steps = 6

  sequence_name = "fi_original"
  [wid, ht] = [1920, 1080]

  img_fold_path = os.path.join('/Users/sfrolov/master-thesis/img/', sequence_name)
  rolo_out_path = os.path.join('/Users/sfrolov/master-thesis/data/ROLO_DATA', sequence_name, 'rolo_out/')

  paths_imgs = utils.load_folder(img_fold_path)

  for i in range(len(paths_imgs) - num_steps):
    id = i + 1
    test_id = id + num_steps - 2  # * num_steps + 1

    path = paths_imgs[test_id]
    img = utils.file_to_img(path)

    if (img is None): break

    rolo_location = utils.find_rolo_location(rolo_out_path, test_id)
    rolo_location = utils.locations_normal(wid, ht, rolo_location)
    print(rolo_location)

    frame = utils.debug_location(img, rolo_location)

    utils.createFolder(os.path.join('output/frames/', sequence_name))
    frame_name = os.path.join('output/frames/', sequence_name, str('{0:04}'.format(test_id)) + '.jpg')
    print(frame_name)
    cv2.imwrite(frame_name, frame)

if __name__ == '__main__':
  main(sys.argv)
