import os
import tqdm
from face_masker import FaceMasker
from skimage.io import imsave


if __name__ == "__main__":
    is_aug = False

    root = "../../data/masks/"
    folder_face = root + "face"
    folder_mask = root + "mask"
    folder_land = root + "landmark"

    face_masker = FaceMasker(is_aug)

    for i in tqdm.trange(1, 45):
        land_file = folder_land + f"/{i:02}_landmark.txt"
        face_file = folder_face + f"/{i:02}.png"
        mask_file = folder_mask + f"/{i:02}.png"
        mask_UV, pos_map = face_masker.get_mask_texture(face_file, mask_file, land_file)
        
        texture_file = os.path.join(root, f"texture/{i:02}.png")
        imsave(texture_file, mask_UV)
        # pos_map_file = os.path.join(root, f"texture/{i:02}_pos_map.png")
        # imsave(pos_map_file, pos_map)