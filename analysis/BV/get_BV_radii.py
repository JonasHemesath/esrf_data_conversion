import argparse
from generate_BV_data import BVDataGenerator

def main(args):
    bv_data_generator = BVDataGenerator(args.brain_regions_path, args.BV_path, args.brain_regions_mip)
    skeleton = bv_data_generator.get_skeleton(label=1)
    
    for vertex in skeleton.vertices:
        print(vertex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the radii of the BV")
    parser.add_argument("--brain_regions_path", type=str, help="Path to the brain regions file")
    parser.add_argument("--BV_path", type=str, help="Path to the BV file")
    parser.add_argument("--brain_regions_mip", type=int, help="MIP level of the brain regions data")
    parser.add_argument("--vertex_coors", type=str, help="Coordinates of the vertex to get the radius for, in the format 'x,y,z'")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    args = parser.parse_args()
    main(args)