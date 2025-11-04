from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, pairs_from_exhaustive
import pycolmap
import argparse
import pathlib
from pixsfm.refine_hloc import PixSfM
import math
import numpy as np
import scipy.spatial.transform
import json

def run_pix_sfm(path_to_image_dir, path_to_working_directory, conf, initial_recons=None):
    path_to_list_of_image_pairs=pathlib.Path(path_to_working_directory / "pairs-netvlad.txt")
    path_to_keypoints=pathlib.Path(path_to_working_directory / "feats-superpoint-n4096-r1600.h5")
    path_to_matches=pathlib.Path(path_to_working_directory / "feats-superpoint-n4096-r1600_matches-superglue_pairs-netvlad.h5")
    # print(conf)
    refiner = PixSfM(conf=conf)
    # print(refiner.conf)
    tmp_out_dir = Path("/tmp/recons")
    tmp_out_dir.mkdir(exist_ok=True)

    if initial_recons is not None:
        initial_recons.write(tmp_out_dir)
        reference_model_path = tmp_out_dir
    else:
        reference_model_path = None
    model, debug_outputs = refiner.triangulation(
        path_to_working_directory,
        path_to_image_dir,
        path_to_list_of_image_pairs,
        path_to_keypoints,
        path_to_matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reference_model_path=reference_model_path
    )
    pass

def convert_fov_to_focal_length(fov, width):
    return (width / 2) / (math.tan(fov/2/math.pi))

def recons_from_transforms(transforms, width):
    total_cams = len(transforms)
    rec = pycolmap.Reconstruction()
    for cam_idx in range(total_cams):
        fov = transforms[str(cam_idx)]["fov"]
        to_world = np.array(transforms[str(cam_idx)]["to_world"])
        from_world = np.linalg.inv(to_world)
        focal_length = convert_fov_to_focal_length(fov, width)
        cx, cy = width/2, width/2
        params = np.array([focal_length, focal_length, cx, cy], dtype=np.float32)
        cam = pycolmap.Camera(model="PINHOLE", width=width, height=width, params=params)
        tvec = from_world[:3, 3:4] # 3x1
        qmat = from_world[:3, :3]
        qvec = scipy.spatial.transform.Rotation.from_matrix(qmat).as_quat().expand_dims(axis=-1)
        img = pycolmap.Image(f"{cam_idx}.png", points2D=[], tvec=tvec, qvec=qvec, camera_id=cam_idx, id=cam_idx)
        rec.add_camera(cam)
        rec.add_image(img)

    return rec
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=pathlib.Path)
    parser.add_argument("output_name", type=str)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--transforms_path", type=str, default=None)
    parser.add_argument("--width", type=str, default=512)
    args = parser.parse_args()
    image_list = [f.name for f in args.images.glob("*.png")]
    outputs = Path('outputs') / args.output_name
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superglue']
    outputs.mkdir(exist_ok=True)
    pairs_from_exhaustive.main(sfm_pairs, image_list)
    feature_path = extract_features.main(feature_conf, args.images, outputs)

    

    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs)
    if args.transforms_path is not None:
        transforms = json.load(open(args.transforms_path, "rt"))
        recons = recons_from_transforms(args.transforms_path, args.width)
    else:
        recons = None
    run_pix_sfm(args.images, outputs, args.conf, initial_recons=recons) 
    pass

if __name__ == "__main__":
    main()
    pass