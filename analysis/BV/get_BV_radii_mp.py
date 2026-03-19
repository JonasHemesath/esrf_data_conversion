import argparse

import subprocess

from generate_BV_data import BVDataGenerator

def main(args):
    bv_data_generator = BVDataGenerator(args.brain_regions_path, args.BV_path, args.brain_regions_mip)
    skeleton = bv_data_generator.get_skeleton(label=1)
    for p_i in range(args.num_processes):
        subprocess.Popen(["python", "get_BV_radii_mp.py", "--brain_regions_path", args.brain_regions_path, "--BV_path", args.BV_path, "--brain_regions_mip", str(args.brain_regions_mip), "--vertex_coors", args.vertex_coors, "--output_file", f"{args.output_file}_process_{p_i}.txt"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the radii of the BV")
    parser.add_argument("--brain_regions_path", type=str, help="Path to the brain regions file")
    parser.add_argument("--BV_path", type=str, help="Path to the BV file")
    parser.add_argument("--brain_regions_mip", type=int, help="MIP level of the brain regions data")
    parser.add_argument("--num_processes", type=int, help="Number of processes to use for parallel processing")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    args = parser.parse_args()