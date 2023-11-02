# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import os
import argparse
import json
import pvmcuda_pkg.sequence_learner as sequence_learner
import pvmcuda_pkg.data as data
import pvmcuda_pkg.data_carla as data_carla
import pvmcuda_pkg.datasets as datasets
import pvmcuda_pkg.manager as manager
import pvmcuda_pkg.readout as readout
import pvmcuda_pkg.synthetic_data as synthetic_data

def execute():
    parser = argparse.ArgumentParser(description="Description",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-a", "--augment", help="Augment factor", type=str, default="0")
    parser.add_argument("-L", "--load", help="Load a pre tgrained model", type=str, default="")
    parser.add_argument("-S", "--spec", help="Specification file name (file in .json format)", type=str, default="")
    parser.add_argument("-o", "--override_spec", help="Specification file name (file in .json format)", type=str, default="")
    parser.add_argument("-D", "--Display", help="Set display onn/off", action="store_true")
    parser.add_argument("-b", "--snapshot", help="Run some experiments", action="store_true")
    parser.add_argument("-e", "--experiment", help="Drop a snapshot", action="store_true")
    parser.add_argument("-t", "--test", help="Print labels for which there is ground truth", action="store_true")
    parser.add_argument("-s", "--synthetic", help="Use synthgetic stimuli", action="store_true")
    parser.add_argument("-r", "--skip_readout", help="Do not instantiate readout object", action="store_true")
    parser.add_argument("-d", "--dataset", help="Name od the dataset to use", type=str, default="")
    parser.add_argument("-f", "--file", help="Data file to be loaded (if dataset not given)", type=str, default="")
    parser.add_argument("-c", "--camvid", help="", type=str, default="")
    parser.add_argument("-p", "--path", help="Path to the data/movie files", type=str, default="")
    parser.add_argument('-O', '--options', type=json.loads,
                        help="Option dictionary (as described above) given in json form '{\"key1\": \"value1\"}\'.",
                        default='{}')
    block_size = 4
    args = parser.parse_args()
    if args.spec != "" or args.load != "":
        if args.spec != "":
            name = os.path.splitext(os.path.basename(args.spec))[0]
            specs = json.load(open(args.spec, "r"))
            SequenceModel = sequence_learner.PVM_object(specs=specs, name=name)
            if args.skip_readout:
                ReadoutModel = None
            else:
                if args.camvid != "" or args.dataset.startswith("camvid") or args.dataset.startswith("carla"):
                    ReadoutModel = readout.Readout(SequenceModel, representation_size=150, heatmap_block_size=2*block_size)
                else:
                    ReadoutModel = readout.Readout(SequenceModel)
        if args.load != "":
            SequenceModel = sequence_learner.PVM_object(specs=None)
            SequenceModel.load(args.load)
            if args.override_spec != "":
                SequenceModel.specs = json.load(open(args.override_spec, "r"))

            if args.skip_readout:
                ReadoutModel=None
            else:
                ReadoutModel = readout.Readout()
                if not ReadoutModel.load(args.load):
                    ReadoutModel = readout.Readout(SequenceModel)
                else:
                    ReadoutModel.set_pvm(SequenceModel)
        xx = SequenceModel.get_input_shape()[0]
        if args.dataset != "":
            if args.dataset.startswith("camvid"):
                Data = data.CamVidDataProvider([args.path + x for x in datasets.sets[args.dataset]], xx, xx, block_size=block_size,
                                               blocks_x=2*int(SequenceModel.specs['layer_shapes'][0]),
                                               blocks_y=2*int(SequenceModel.specs['layer_shapes'][0]),
                                               augment=int(args.augment))
            elif args.dataset.startswith("carla"):
                Data = data_carla.CarlaVideoProvider([args.path + x for x in datasets.sets[args.dataset]], xx, xx,
                                               block_size=block_size,
                                               blocks_x=2 * int(SequenceModel.specs['layer_shapes'][0]),
                                               blocks_y=2 * int(SequenceModel.specs['layer_shapes'][0]),
                                               augment=int(args.augment))
            else:
                Data = data.ZipCollectionDataProvider(
                    [args.path + x for x in datasets.sets[args.dataset]], xx, xx)
        elif args.file != "":
            if args.file.endswith("MOV") or args.file.endswith("mov") or args.file.endswith("avi") or args.file.endswith("mp4"):
                Data = data.MovieDataProvider(os.path.join(args.path,args.file), xx, xx)
            elif args.file.endswith("zip") or args.file.endswith("ZIP"):
                Data = data.ZipDataProvider(os.path.join(args.path,args.file), xx, xx)
        elif args.camvid != "":
                Data = data.CamVidSingleDataProvider(os.path.join(args.path, args.camvid), xx, xx, block_size=block_size, blocks_x=int(SequenceModel.specs['layer_shapes'][0]),
                                               blocks_y=int(SequenceModel.specs['layer_shapes'][0]))
        elif args.synthetic:
            Data = synthetic_data.SyntheticDataProvider(xx, xx, block_size=block_size, blocks_x=int(SequenceModel.specs['layer_shapes'][0]),
                                               blocks_y=int(SequenceModel.specs['layer_shapes'][0]))

        else:
            raise Exception("No data file given")
        if args.test:
            Data.set_attr("test", True)
        ModelManager= manager.PVMManager(PVMObject=SequenceModel, 
                                         DataProvider=Data, 
                                         ReadoutObject=ReadoutModel, 
                                         snapshot=args.snapshot)
        ModelManager.do_display = args.Display
        ModelManager.run(steps=100000000)
        ModelManager.stop()

# Example command:
# python run.py  -S ./model_zoo/med.json -D -d face_training -p ~/old_PVMdata/
# Assuming that the converted to zip PVM data is in ~/old_PVMdata/
if __name__ == "__main__":
    execute()