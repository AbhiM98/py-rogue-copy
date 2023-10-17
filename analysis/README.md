# Analysis Tools

A python module for analyzing data produced by the `ground_data_processing` module

## To Do (in order of priority):
- Fix pulling images from s3, currently broken


## High Level Breakdown

Corn fields were staked by Bayer, these stakes are treated as ground truth and are how we "learn" what is and isn't a rogue. The `ground_data_processing` module produces segmented images for analysis, see that directory for more details. 

The segmented images contain a plethora of leaves whose masks are used to compute length, width, and area. These values are extracted through some image analysis and stored as a Stats class. A statistical voting classifier is built, using the ground truth, to identify rogue corn plants. The model is validated on hybrid rogue corn plants classified by-eye by Sentera GIS Analysts.

### The Process

Start by building a list of files to pull from s3 using the following script:
```
python scripts/build_s3_keys_json.py --field <field_name> --subfield <subfield_name> --planting <planting_number> --date <date_of_analysis> --row <row_name> --row_pass <row_pass> --output <output_name> 
```
The script above will produce a json file containing a list of s3 keys to pull from s3. The output file name is structured as `<field_name>_<subfield_name>_<planting_number>_<date_of_analysis>_Row<row_num>_s3_keys.json`. Check the `jsons/` directory for examples. The script will also print the number of `segm.json` files and `rogues.json` files it found and attempt to align them. If the numbers don't match, it is best to inspect for errors.

The `run_analysis.py` will pull everything from it needs from s3 to local, provided the correct json file from the previous step.
The next step is to create a threshold to exclude leaves from neighboring rows that may wander into the camera's field of view. This is done by running the following scripts:
```
python scripts/run_analysis.py --dir /path/to/directory/containing/ds/splits/
python scripts/analyze_data.py -d /path/to/directory/containing/ds/splits/measurements_df.gz --threshold
```

Once done, inspect the `centroid_hist.png` plot and use the lower right histogram to determine a threshold. This threshold is used to filter out leaves that are not in the same row.

Now you can extract the features from the masks:
```
python scripts/run_analysis.py --dir /path/to/directory/containing/ds/splits/ --threshold <min-threshold> --threshold <max-threshold>
```

The script above takes two options: the directory containing your ds_splits subdirectories, and a threshold.
It will produce a gzipped pickle file containing a dataframe that stores the following information:

 - "name" : image name in format DS XXX_XXX
 - "image-id" : image id in format XXX
 - "Stats-length-nadir" : Stats object containing length data for nadir leaves
 - "Stats-width-nadir" : Stats object containing width data for nadir leaves
 - "Stats-area-nadir" : Stats object containing area data for nadir leaves
 - "Centroids-nadir" : list of centroids for nadir leaves in format ((x,y), weight)
 - "Stats-length-oblique" : Stats object containing length data for oblique leaves
 - "Stats-width-oblique" : Stats object containing width data for oblique leaves
 - "Stats-area-oblique" : Stats object containing area data for oblique leaves
 - "Stats-mask-pct-in-upper-thirty-oblique" : Stats object containing mask pct data for oblique leaves
 - "Stats-mask-tallest-oblique" : Stats object containing mask tallest data for oblique leaves
 - "cumulative-distribution-mask-y-values" : normalized cumulative distribution of masked pixels in each row
 - "Centroids-oblique" : list of centroids for oblique leaves in format ((x,y), weight)
 - "rogue" : label for corn plants ["normal", "Hybrid Rogue White", "Hybrid Rogue Yellow", "Mix Rogue White", etc.]
 - "dataset" : dataset name in format <field>_<date>_Row<row_num> (mostly useful when processing many datasets at once)

Finally, the data is processed and a model is built. Model building is automated, but often requires some QA. Model building and validation is done using the `scripts/analyze_data.py` script.

To build a static model:
```
python scripts/analyze_data.py --dataset /path/to/directory/containing/ds/splits/measurements_df.gz --mode 'build` --output_dir <output_directory>
``` 
if you want to log this to the MLFlow page, provide the `--log` flag. This will produce a model saved to `models/<output_directory>_model.json`, as well as printing out a validation report and plotting the features used to build the model. Additionally, the model and validation report are saved to an MLFlow experiment here: http://ec2-35-170-127-146.compute-1.amazonaws.com:5000/#/experiments/45

To build a dynamic model:
```
python scripts/analyze_data.py --dataset /path/to/directory/containing/ds/splits/measurements_df.gz --mode 'dynamic` --output_dir <output_directory> 
```

The analyze_data.py script has many additional options. I'll list a few of them here, but to see the full list go check out the script:

    - `--dataset` : path to dataset to analyze
    - `--mode` : mode to run in ['build', 'all']
    - `--output_dir` : directory to save model and plots to
    - `--threshold` : flag to generate threshold plots
    - `--no_plot` : flag to turn off plotting
    - `--log` : flag to log to MLFlow experiment
    - `--no_oblique` : flag to turn off oblique leaf analysis

## Advanced Features:

You can load the interactive matplotlib gui for most plots by opening them as a pickle file on a local machine.
Start by opening a python environment
```
python
```
then use following commands to load the interactive plot:
```,;''''''''''''''''''/';/'.;
import pickle
figx = pickle.load(open(<pickle-file>,'rb'))
figx.show()
```
## Detailed Script Usage:

detailed instructions on how to use each script.

### scripts/analyze_data.py

### scripts/build_s3_keys_json.py

The expected format for a json file is the following:
```
{
    "s3_keys_by_plot": [
        {
            "segm_s3_key_nadir": "/path/to/segm-nadir.json",
            "segm_s3_key_oblique": "/path/to/segm-oblique.json",
            "segm_s3_key_bottom": "/path/to/segm-bottom.json",
            "rogue_label_s3_key": "path/to/labels.json",
        },
        {
            "segm_s3_key_nadir": "/path/to/segm-nadir.json",
            "segm_s3_key_oblique": null,
            "segm_s3_key_bottom": null,
            "rogue_label_s3_key": null,
        }
    ]
}
```

The `s3_keys_by_plot` key must have at least one full entry. Each key must be specified, if no path exists for the object then provide `null` as demonstrated in the second entry in the example above. If your json file is generated with `scripts/build_s3_keys_json.py` it will automatically have the right format, but it may be worth checking that each file in each entry is for the correct DS Split. You must, at a minimum, specify the nadir segmentation. Without a nadir segmentation file the `scripts/run_analysis.py` script will crash.


### scripts/run_analysis.py

Options:
- `--input_file`: input JSON (usually made using scripts/build_s3_keys_json.py) that specifies which files to use.
- `--output_dir`: user specified output directory. If this is left empty the script will attempt to determine where to write the output based on the folder structure of the first file in the `--input_file`. The program will try to look for the `DS Splits` folder and write to that path.
- `--threshold`: centroid threshold for removing leaves from neighboring rows (left and right). Best practice is to provide both the min and max threshold values as `--threshold <min-threshold> --threshold <max-threshold>`. 
- `--update_rogues`: (flag) update the measurements file with the new rogue labels (advanced option, only relevant if the rogue labels on s3 have changed, but the images have not)
- `--overwrite`: (flag) overwrites local files with new files from s3 (this is only relevant if the files in your `--input_file` are on s3, the program will determine if they exist on s3 and if they do not it will not bother trying to pull them)
- `--no_s3`: (flag) prevent writing measurement files to s3

Usage cases:
- If your segm files are on s3:
    - First, build your input json with `scripts/build_s3_keys_json.py`, then run the following:  
        - `python scripts/run_analysis.py --input_file /path/to/input/file.json --threshold <min-threshold> --threshold <max-threshold> --overwrite`
    - The overwrite flag here is optional, but it is recommended you provided it every time.
- If your segm files are local:
    - First, build your input json manually with the format specified under the `scripts/build_s3_keys_json.py` section.
    - Then, run the following:
        - `python scripts/run_analysis.py --input_file /path/to/input/file.json --output_dir /path/to/output/directory/ --threshold <min-threshold> --threshold <max-threshold> --no_s3`
    - The program will likely break if your `--output_dir` is not specified.
    - The `--no_s3` flag prevents the program from trying to write to s3, which can cause a mess on s3 if you are not careful.

