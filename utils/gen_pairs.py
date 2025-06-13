import os
import random
from argparse import ArgumentParser


def gen_pairs(dataset: str, root_dir: str) -> None:
    """Generate pairs for the given dataset"""
    if dataset == "celeba_hq":
        root = "data/celeba_hq"
        identities = os.path.join(root, "identities.txt")
        dodging = os.path.join(root, "dodging.txt")
        impersonation = os.path.join(root, "impersonation.txt")

        with open(identities) as f:
            lines = f.readlines()[1:]
            id_to_images = {}

            for line in lines:
                line = line.strip().split(",")
                img_list = id_to_images.get(line[1], [])
                img_list.append(line[0])
                id_to_images[line[1]] = img_list

            # create dodging list
            with open(dodging, "w") as out:
                face_pairs = []
                used_pairs = set()
                while len(face_pairs) < 1000:
                    # choose a random identity
                    # and sample two images from that identity
                    # to create a dodging pair
                    id1 = random.choice(list(id_to_images.keys()))
                    if len(id_to_images[id1]) >= 2:
                        face1, face2 = random.sample(id_to_images[id1], 2)
                        face1_landmark = os.path.join(root_dir, f"{face1}_landmark.txt")
                        face2_landmark = os.path.join(root_dir, f"{face2}_landmark.txt")
                        if (
                            os.path.exists(face1_landmark)
                            and os.path.exists(face2_landmark)
                            and (face1, face2) not in used_pairs
                        ):
                            used_pairs.add((face1, face2))
                            face_pairs.append((face1, face2))
                            out.write(f"{face1},{face2}\n")

            # create impersonation list
            with open(impersonation, "w") as out:
                face_pairs = []
                used_pairs = set()
                while len(face_pairs) < 1000:
                    # choose two random identities
                    # and sample one image from each identity
                    # to create an impersonation pair
                    id1, id2 = random.sample(list(id_to_images.keys()), 2)
                    face1 = random.choice(id_to_images[id1])
                    face2 = random.choice(id_to_images[id2])
                    face1_landmark = os.path.join(root_dir, f"{face1}_landmark.txt")
                    face2_landmark = os.path.join(root_dir, f"{face2}_landmark.txt")
                    if (
                        os.path.exists(face1_landmark)
                        and os.path.exists(face2_landmark)
                        and (face1, face2) not in used_pairs
                    ):
                        used_pairs.add((id1, id2))
                        face_pairs.append((face1, face2))
                        out.write(f"{face1},{face2}\n")

    else:  # lfw
        pairs = os.path.join(root_dir, "pairs.csv")
        image_path = os.path.join(root_dir, "lfw-deepfunneled")
        save_dir = "data/lfw"
        dodging_list = os.path.join(save_dir, "dodging.txt")
        impersonation_list = os.path.join(save_dir, "impersonation.txt")

        with open(pairs) as f:
            lines = f.readlines()[1:]

            dodging_paris = []
            impersonation_pairs = []

            # Split pairs into dodging and impersonation
            for i, line in enumerate(lines):
                line = line.strip().split(",")
                if line[3] == "":  # dodging
                    face1_landmark = os.path.join(
                        image_path,
                        f"{line[0]}/{line[0]}_{int(line[1]):04d}_cropped_landmark.jpg",
                    )
                    print(face1_landmark)
                    face2_landmark = os.path.join(
                        image_path,
                        f"{line[0]}/{line[0]}_{int(line[2]):04d}_cropped_landmark.jpg",
                    )
                    if os.path.exists(face1_landmark) and os.path.exists(
                        face2_landmark
                    ):
                        dodging_paris.append(
                            f"{line[0]},{int(line[1]):04d},{line[0]},{int(line[2]):04d}\n"
                        )
                else:  # Impersonation pairs
                    face1_landmark = os.path.join(
                        image_path,
                        f"{line[0]}/{line[0]}_{int(line[1]):04d}_cropped_landmark.jpg",
                    )
                    face2_landmark = os.path.join(
                        image_path,
                        f"{line[2]}/{line[2]}_{int(line[3]):04d}_cropped_landmark.jpg",
                    )
                    if os.path.exists(face1_landmark) and os.path.exists(
                        face2_landmark
                    ):
                        impersonation_pairs.append(
                            f"{line[0]},{int(line[1]):04d},{line[2]},{int(line[3]):04d}\n"
                        )

            # Randomly select 1000 pairs from each category
            random.shuffle(dodging_paris)
            random.shuffle(impersonation_pairs)
            dodging_paris = dodging_paris[:1000]
            impersonation_pairs = impersonation_pairs[:1000]
            # Write to output files
            with open(dodging_list, "w") as out:
                out.writelines(dodging_paris)
            with open(impersonation_list, "w") as out:
                out.writelines(impersonation_pairs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["celeba_hq", "lfw"],
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="The directory of the images.",
    )

    random.seed(42)

    args = parser.parse_args()

    gen_pairs(args.dataset, args.root_dir)
