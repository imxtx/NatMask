from face_masker import FaceMasker

if __name__ == "__main__":
    # is_aug = False
    # image_path = "Data/test-data/031.jpg"
    # face_lms_file = "Data/test-data/031_landmark.txt"
    # for i in range(1, 45):
    #     template_name = f"{i:02}.png"
    #     masked_face_path = f"/home/txxie/dataset/mask3.0/test256/{i:02}.png"
    #     face_lms_str = open(face_lms_file).readline().strip().split(" ")
    #     face_lms = [float(num) for num in face_lms_str]
    #     face_masker = FaceMasker(is_aug)
    #     face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)

    # For paper figures
    is_aug = False
    image_path = "Data/for_paper/attacker.png"
    face_lms_file = "Data/for_paper/attacker_landmark.txt"
    for i in range(26, 27):
        template_name = f"{i:02d}.png"
        masked_face_path = f"Data/for_paper/attacker_masked_{i:02d}.png"
        face_lms_str = open(face_lms_file).readline().strip().split(" ")
        face_lms = [float(num) for num in face_lms_str]
        face_masker = FaceMasker(is_aug)
        face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)
