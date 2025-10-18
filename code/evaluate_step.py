import numpy as np

import json
import src.utils.eval_modules as eval_m
import src.utils.io.gt as gt
import os

device = "cuda:0"

eval_fn_dict = {
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    # "mrrpe_ho": eval_m.eval_mrrpe_ho_right,
    # "cd_f_ra": eval_m.eval_cd_f_ra,
    # "cd_f_right": eval_m.eval_cd_f_right,
}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_p", type=str, default="")
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--mvs_root", type=str, default="")
    parser.add_argument("--object_mesh_f", type=str, default="")
    parser.add_argument("--MANO_f", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--only_eval_hand", default=False, action="store_true")
    
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args

def main():
    from tqdm import tqdm
    args = parse_args()
    import src.utils.io.ours as ours

    data_pred = ours.load_data_step(
        seq_name=args.seq_name,
        data_root=args.data_root,
        mvs_root=args.mvs_root,
        object_mesh_f=args.object_mesh_f,
        ckpt_p=args.sd_p,
        MANO_f=args.MANO_f,
    )
    data_pred["out_dir"] = args.out_dir
    data_gt = gt.load_data_diff_object(
        full_seq_name=args.seq_name,
        mvs_root=args.mvs_root,
        debug=args.debug,
    )
        
    
    seq_name = data_pred["full_seq_name"]
    out_p = args.out_dir
    os.makedirs(out_p, exist_ok=True)
    if not args.only_eval_hand:
        eval_fn_dict["icp"] = eval_m.eval_icp_first_frame
        eval_fn_dict["cd_f_right"] = eval_m.eval_cd_f_right

    print("------------------")
    print("Involving the following eval_fn:")
    for eval_fn_name in eval_fn_dict.keys():
        print(eval_fn_name)
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(eval_fn_dict.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + "/metric.json"
    npy_path = out_p + "/metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm**2), F-score (percentage), MPJPE (mm)")

    # Save the mean_metrics dictionary to a JSON file with indentation
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")


if __name__ == "__main__":
    main()
