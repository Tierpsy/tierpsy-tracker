#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Running batch processing on all videos in a directory with specified parameters.
IMPORTANT:
- check python_exec path and script_path in the code below

Usage:
activate your environment first:
conda activate {tierpsy env name}

Example to just get feature summaries, No featureN.hdf5 file
python tierpsy_param_sweep.py --input_dir my_sample_path/MaskedVideos --output_dir my_sample_path/Results --json_file samples/batch_processing_parameter_file.json --del_features false --threads 8

Example to just get feature summaries and featureN.hdf5 files, might use a huge amount of space!  
python tierpsy_param_sweep.py --input_dir my_sample_path/MaskedVideos --output_dir my_sample_path/Results --json_file samples/batch_processing_parameter_file.json --del_features true --threads 8


Main workflow:
1. Find all masked video files in the input directory.
2. For each combination of parameters, process all videos in parallel using Tierpsy original ProcessLocal script.
3. After processing all videos for a parameter combination, calculate feature summaries.
4. Save the summary files with parameter combination identifiers and optionally delete intermediate feature files.



Arguments:

Author: Hossein Khabbaz
Date: 2025-8
"""

import os
import argparse
import time
import yaml
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from tierpsy.summary.collect import calculate_summaries

# initials
path = os.path.abspath(__file__)
processLocal_path = path.replace('extras/tierpsy_param_sweep.py', 'processing/ProcessLocal.py')

def find_hdf5_files(input_dir, extensions=(".hdf5")):
    for dirpath, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(extensions):
                yield Path(dirpath) / f

def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}\n{e}")

def main(args):
    # Convert input_dir and tmp_dir to Path objects and resolve them
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    select_feat = args.select_feat
    threads = args.threads
    # tmp_dir = Path(args.tmp_dir).resolve()
    # Parameter grid
    params_options_dict = {
        'dwelling_threshold': [5],# Decided on 5 based on MMP data
        'roaming_threshold': [25],#[10,15, 20, 25, 30, 35, 40],
        'sprinting_threshold': [100],#[100, 120, 140, 160, 180, 200],
        'angular_velocity_threshold': [0.3],#[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #'smoothing_window_sec': [1.2],
        'delta_frames': [75], #[ 50, 75, 125]
        'min_compactness':[0.4], #[0.3, 0.4, 0.5]
        'compactness_window':[2] 
    }

    param_names = list(params_options_dict.keys())
    param_combos = list(product(*(params_options_dict[k] for k in param_names)))
    hdf5_files = list(find_hdf5_files(input_dir))
    print(f"Found {len(hdf5_files)} masked video files.")

    # create feature summaries directory called "FeatureSummaries" if it does not exist
    feature_summaries_dir = output_dir.parent / "FeatureSummaries"
    feature_summaries_dir.mkdir(parents=True, exist_ok=True)

    param_combo_file_name = 'parameters_combo.yaml'
    # find the absolute path of the extras directory in the tierpsy-tracker package based on the location of the script_path argument
    script_path = Path(args.script_path).resolve()
    tierpsy_tracker_dir = script_path.parent.parent.parent
    extras_dir = tierpsy_tracker_dir / 'tierpsy' / 'extras'
    # create the absolute path of the parameters_combo.yaml file in the extras directory
    PARAM_COMBO_FILE = extras_dir / param_combo_file_name


    total_jobs = len(param_combos) * len(hdf5_files)
    job_counter = 0

    for combo in param_combos:
        param_dict = dict(zip(param_names, combo))
        param_str = "_".join(str(v) for v in combo)
        results_base = Path(output_dir)
        masks_base = Path(input_dir)
        # tmp_results_base = Path(tmp_dir)
        # tmp_masks_base = Path(tmp_dir) / f"MaskedVideos_{param_str}"

        # Write the param file ONCE for this combo
        with open(PARAM_COMBO_FILE, 'w') as f:
            yaml.dump(param_dict, f)

        # Prepare commands for all videos for this combo
        jobs = []
        for masked_file in hdf5_files:
            rel_path = masked_file.relative_to(input_dir)

            mask_dir = (masks_base / rel_path.parent).resolve()
            result_dir = (results_base / rel_path.parent).resolve()
            tmp_mask_dir = ''
            tmp_result_dir = ''
        
            command = [
                args.python_exec, args.script_path,
                str(masked_file),
                '--masks_dir', str(mask_dir),
                '--results_dir', str(result_dir),
                '--tmp_mask_dir', str(tmp_mask_dir),
                '--tmp_results_dir', str(tmp_result_dir),
                '--json_file', str(Path(args.json_file).resolve()),
                '--analysis_checkpoints'
            ] + args.analysis_checkpoints.split()
            jobs.append(command)

        # Parallelize only over videos for this combo
        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(run_command, cmd) for cmd in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Job failed: {e}")
                job_counter += 1
                print(f"Completed {job_counter}/{total_jobs}")        
        # Calculate summaries after processing all videos for this combo
        calculate_summaries(
            root_dir = results_base,
            feature_type = 'tierpsy',
            summary_type = 'plate',
            is_manual_index = False,
            abbreviate_features = False,
            dorsal_side_known = False,
            time_windows='0:end',
            time_units = None,
            select_feat = select_feat,
            keywords_include = '',
            keywords_exclude = '',
            n_parallel = threads
            )
        # copy the summary file to a feature summaries directory with param_str in the name of the file
        # find the summary file by searching for files starting with "features_summary"
        summary_files = list(results_base.glob("**/features_summary*.csv"))
        for summary_file in summary_files:
            if summary_file.is_file():
                new_summary_file = feature_summaries_dir / f"{summary_file.stem}_{param_str}.csv"
                summary_file.rename(new_summary_file)
                print(f"Moved summary file to {new_summary_file}")

        # delete hdf5 features files (basename_featuresN.hdf5, if user requested) and csv feature summaries and filenames files in the Results directory
        for result_file in results_base.glob("**/*_featuresN.hdf5"):
            if result_file.is_file():
                if args.del_features.lower() == "true":
                    result_file.unlink()
                    print(f"Deleted {result_file}")
                else:
                    # rename feature file to include param_str
                    new_result_file = result_file.parent / f"{result_file.stem}_{param_str}.hdf5"
                    result_file.rename(new_result_file)
                    print(f"Renamed feature file to {new_result_file}")
        # delete all summary files in the Results directory
        for summary_file in results_base.glob("**/features_summary*.csv"):
            if summary_file.is_file():
                summary_file.unlink()
                print(f"Deleted {summary_file}")
        for filename_file in results_base.glob("**/filenames*.csv"):
            if filename_file.is_file():
                filename_file.unlink()
                print(f"Deleted {filename_file}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument('--tmp_dir', type=str, required=True)
    parser.add_argument('--python_exec', type=str, default="python")
    parser.add_argument('--script_path', type=str, default=processLocal_path)
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--del_features', type=str, default="True", choices=['true', 'false'],
                        help="Whether to delete the features files after processing. Default is true.")
    parser.add_argument('--analysis_checkpoints', type=str, default="FEAT_INIT FEAT_TIERPSY")
    parser.add_argument('--select_feat', type=str, default="tierpsy_level_4", choices=['tierpsy_level_0','tierpsy_level_1', 'tierpsy_level_2', 'tierpsy_level_3', 'tierpsy_level_4'],
                        help="Select which feature set to use for summary calculation. Default is tierpsy_level_4.")
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    main(args)