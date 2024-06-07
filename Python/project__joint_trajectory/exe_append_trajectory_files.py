#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import shutil
import os
import re
import argparse
from pathlib import Path

def copy_files(input_folder_regex, output_folder_path, regex):
    list_input_traj_files = sorted(glob.glob(input_folder_regex))
    list_output_traj_files = sorted(glob.glob(output_folder_path))

    output_folder_path_regex = Path(output_folder_path) / Path(regex)
    list_output_traj_files = sorted(glob.glob(output_folder_path_regex.as_posix()))
    last_index = -1
    if list_output_traj_files:
        last_file = os.path.basename(list_output_traj_files[-1])
        # print(f"last_file={last_file}")

        last_index = int(re.search('^\d{5}', last_file).group())
        # print(f"last_index={last_index}")

    start_index = last_index + 1
    print(f"start_index={start_index}")

    for xml_file in list_input_traj_files:
        input_filename = os.path.basename(xml_file)
        current_index = int(re.search('^\d{5}', input_filename).group())
        end_filename = re.search('_\S+.xml', input_filename).group()
        output_filename = "{:05d}".format(start_index+current_index) + end_filename
        # print(f"current_index={current_index} ; output_filename={output_filename}")

        output_filepath = os.path.join(output_folder_path, output_filename)
        # print(f"xml_file={xml_file} ; output_filepath={output_filepath}")
        shutil.copyfile(xml_file, output_filepath)

def main():
    parser = argparse.ArgumentParser(description='Append.')
    parser.add_argument("--input", help='Input folder without regex.', nargs='+', default=[])
    parser.add_argument("--output", help='Output folder.')
    parser.add_argument("--regex", default="*.xml", help='Regex.')
    args = parser.parse_args()

    input_folders = args.input
    regex = args.regex
    # print(f"input_folders={input_folders}")
    # print(f"regex={regex}")

    for input_folder_str in input_folders:
        input_folder = Path(input_folder_str)
        print(f"\ninput_folder={input_folder}")

        input_folder_regex = input_folder / Path(args.regex)
        output_folder_path = Path(args.output)

        # print(f"input_folder_regex={input_folder_regex}")
        # print(f"output_folder_path={output_folder_path}")

        if input_folder.is_dir() and output_folder_path.is_dir():
            copy_files(input_folder_regex.as_posix(), output_folder_path.as_posix(), regex)
        else:
            print(f"{input_folder} is not valid or {output_folder_path} is not valid")

if __name__ == "__main__":
    main()