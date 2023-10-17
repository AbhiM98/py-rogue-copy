"""Classes containing parameters used in analysis."""


class AnalysisParameters:
    """Parameters used in analysis."""

    # List of Features to use in analysis
    ANALYSIS_FEATURES = [
        "name",
        "image-id",
        "Stats-length",
        "Stats-width",
        "Stats-area",
        "Centroids",
        "rogue",
    ]
    ANALYSIS_FEATURES_OBLIQUE = [
        "name",
        "image-id",
        "Stats-length-nadir",
        "Stats-width-nadir",
        "Stats-area-nadir",
        "Centroids-nadir",
        "Stats-length-oblique",
        "Stats-width-oblique",
        "Stats-area-oblique",
        "Stats-mask-pct-in-upper-thirty-oblique",
        "Stats-mask-tallest-oblique",
        "cumulative-distribution-mask-y-values",
        "Centroids-oblique",
        "rogue",
    ]

    STATS_KEYS = [
        "name",
        "mean",
        # "median",
        # "skew",
        # "kurtosis",
        "stddev",
        "max",
        "sum_top_10_pct",
    ]

    JSON_SEGM = "segm"
    JSON_AREA = "mask_areas"
    JSON_TRUTH = "unfiltered-rogues"

    LOCAL_PATH_ROOT = "s3_mirror"
    S3_PATH_ROOT = "s3://sentera-rogues-data"

    JSON_PRIMARY_KEY = "s3_keys_by_plot"
    JSON_ROGUE_KEY = "rogue_label_s3_key"
    JSON_SEGM_OBLIQUE_KEY = "segm_s3_key_oblique"
    JSON_SEGM_NADIR_KEY = "segm_s3_key_nadir"
    JSON_SEGM_KEY = "segm_s3_key"

    MASKS = "masks"
    SCORES = "scores"
    AREAS = "areas"
    ROGUE = "rogue"
    IMAGES = "images"
    MASK_AREAS = "mask_areas"
    PLANT_TYPE = "plant_type"
    STAKE_COLOR = "stake_color"
    HYBRID = "Hybrid Rogue"
    NOT_LABELED = "Normal"
    HYBRID_WHITE = "Hybrid Rogue White"
    HYBRID_YELLOW = "Hybrid Rogue Yellow"
    json_rogue = "unfiltered-rogues"
    ALL_MASK_AREAS = "all_mask_areas"
    MASK_DICT = {MASKS: [], AREAS: [], ROGUE: ""}
    SEGM = "segm"
    SEGMENTATION = "segmentation"
    SEGM_NADIR = "segm-nadir"
    SEGM_OBLIQUE = "segm-oblique"
    IMG_ID = "image_id"
    CENTROID_THRESHOLD = [0, 99999]
    SCORE = "score"
    SCORE_THRESHOLD = 0.25

    COMPRESSION_DICT = {"method": "gzip", "compresslevel": 1, "mtime": 1}


class ModelParameters:
    """Parameters used in models."""

    MODEL_PARAMS = [
        "cut",
        "min_viable_cut",
        "window",
    ]

    SIGMA = "sigma_exp"
    INFLUENCE = "influence"
    ACTIVE = "active"
    MIN_VIABLE = "min_viable"
    MODEL_STATS_PARMS = [
        SIGMA,
        INFLUENCE,
        ACTIVE,
    ]  # min viable left out because it's not a default parameter
    CORRELATION_THRESHOLD = 0.45

    MLFLOW_URI = "http://ec2-35-170-127-146.compute-1.amazonaws.com:5000/"
    MLFLOW_EXPERIMENT_NAME = "py_rogues_detection_models"


class PlottingParameters:
    """Parameters used in plotting ."""

    COLOR_LIST = {
        "Navy": "#303948",  # Sentera Navy
        "Green": "#799b3e",  # Sentera Green
        "Gold": "#ddb426",  # Senter Gold
        "Orange": "#ee9821",  # Sentera Orange
        "Beige": "#d0a584",  # Sentera Beige
        "Gray": "#636464",  # Sentera Gray
        "Blue": "#486ab2",  # Sentera Blue
        "Teal": "#309685",  # Sentera Teal
        "Red": "#cd622d",  # Sentera Red-Orange
        "Brown": "#a8844f",  # Sentera Brown
    }

    COLOR_ORDER = [  # ordered colors for plots
        "Navy",  # primary line
        "Orange",  # secondary line
        "Green",
        "Gold",
        "Gray",
        "Red",
        "Blue",
        "Teal",
        "Brown",
        "Beige",
    ]

    COLORS = []
    for x in COLOR_ORDER:
        COLORS.append(COLOR_LIST[x])
    # had to do this because of scoping

    # TODO: Add watermark to plots

    LW_THIN = 1.5
    LW_THICK = 1.75

    LW_VLINE = 0.5
    LSTYLE_DASHED = (0, (5, 10))
    LSTYLE_SOLID = "solid"
    COLOR_VLINE = COLOR_LIST["Teal"]

    LOC_LEGEND = "upper right"

    ADJUST_LEFT = 0.075
    ADJUST_RIGHT = 0.99
    ADJUST_TOP = 0.95
    ADJUST_BOTTOM = 0.10
    ADJUST_BOTTOM_SPLIT = 0.175

    ANNOT_SENTERA = "Sentera"
    ANNOT_INTERNAL = "Internal"


class S3Paths:
    """Paths to S3 data."""

    fields = {
        "Waterman_Strip_Trial": {
            "name": "Waterman_Strip_Trial",
            "num_rows": 30,
        },
        "Williamsburg_Strip_Trial": {
            "name": "Williamsburg_Strip_Trial",
            "num_rows": 30,
        },
        "PF1": {
            "name": "Production Field 1 (Argo North-Home Minor)",
            "num_rows": 5,
            "inference_depths": {
                "field": 0,
                "inference": 1,
                "date": 4,
                "row": 5,
                "segm": 10,
            },
            "video_depths": {
                "field": 0,
                "video": 1,
                "date": 2,
                "row": 3,
                "rogues": 7,
            },
        },
        "FF1": {
            "name": "Foundation Field 1",
            "num_rows": 8,
            "row_names": ["Row 1, 8", "Row 2, 7", "Row 3, 6", "Row 4, 5"],
            "inference_depths": {
                "field": 0,
                "inference": 1,
                "date": 4,
                "row": 5,
                "row_pass": 7,
                "ds_id": 9,
                "segm": 11,
                "img": 12,
            },
            "video_depths": {
                "field": 0,
                "video": 1,
                "date": 2,
                "row": 3,
                "row_pass": 5,
                "ds_id": 7,
                "rogues": 8,
                "img": 9,
            },
        },
        "FF2": {
            "name": "Foundation Field 2 (Dennis Zuber)",
            "num_rows": 16,
            "row_names": [],
            "inference_depths": {
                "field": 0,
                "inference": 1,
                "date": 4,
                "row": 5,
                "row_pass": 7,
                "segm": 11,
                "img": 12,
            },
            "video_depths": {
                "field": 0,
                "video": 1,
                "date": 2,
                "row": 3,
                "row_pass": 5,
                "rogues": 8,
            },
        },
        "FC22_StripTrial_P1": {
            "name": "Farmer City 2022",
            "num_rows": 12,
            "row_names": [
                "Row 1b, 6a",
                "Row 2b, 5a",
                "Row 3b, 4a",
                "Row 4b, 3a",
                "Row 5b, 2a",
                "Row 6b, 1a",
            ],
            "inference_depths": {
                "field": 0,
                "subfield": 1,
                "planting": 2,
                "inference": 3,
                "date": 6,
                "row": 7,
                "row_pass": 9,
                "ds_id": 11,
                "segm": 13,
                "img": 14,
            },
            "video_depths": {
                "field": 0,
                "subfield": 1,
                "planting": 2,
                "video": 3,
                "date": 4,
                "row": 5,
                "row_pass": 7,
                "ds_id": 9,
                "rogues": 10,
                "img": 11,
            },
        },
        "FC22_SmallPlot_P1": {
            "name": "Farmer City 2022",
            "num_rows": 12,
            "row_names": [
                "12, 1",
                "11, 2",
                "10, 3",
                "9, 4",
                "8, 5",
                "7, 6",
            ],
            "inference_depths": {
                "field": 0,
                "subfield": 1,
                "planting": 2,
                "inference": 3,
                "date": 6,
                "row": 7,
                "row_pass": 9,
                "ds_id": 11,
                "segm": 13,
                "img": 14,
            },
            "video_depths": {
                "field": 0,
                "subfield": 1,
                "planting": 2,
                "video": 3,
                "date": 4,
                "row": 5,
                "row_pass": 7,
                "ds_id": 9,
                "rogues": 10,
                "img": 11,
            },
        },
        "FC22_SmallPlot_P2": {
            "name": "Farmer City 2022",
            "num_rows": 12,
            "row_names": [
                "12, 1",
                "11, 2",
                "10, 3",
                "9, 4",
                "8, 5",
                "7, 6",
            ],
            "inference_depths": {
                "field": 0,
                "subfield": 1,
                "planting": 2,
                "inference": 3,
                "date": 6,
                "row": 7,
                "row_pass": 9,
                "ds_id": 11,
                "segm": 13,
                "img": 14,
            },
            "video_depths": {
                "field": 0,
                "subfield": 1,
                "planting": 2,
                "video": 3,
                "date": 4,
                "row": 5,
                "row_pass": 7,
                "ds_id": 9,
                "rogues": 10,
                "img": 11,
            },
        },
    }

    subfields = {
        "SmallPlot": "Small Plot",
        "StripTrial": "Strip Trial",
    }

    planting = {
        "P1": "Planting 1",
        "P2": "Planting 2",
    }

    passes = {
        "A": "Pass A",
        "B": "Pass B",
    }

    years = ["22", "23"]

    paths = {
        "PF1": {
            "6-27": {
                "Row1": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-27/Row 1/DS Splits",
                "Row2": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-27/Row 2/DS Splits",
                "Row3": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-27/Row 3/DS Splits",
                "Row4": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-27/Row 4/DS Splits",
                "Row5": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-27/Row 5/DS Splits",
            },
            "7-05": {
                "Row1": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-05/Row 1/DS Splits",
                "Row2": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-05/Row 2/DS Splits",
                "Row3": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-05/Row 3/DS Splits",
                "Row4": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-05/Row 4/DS Splits",
                "Row5": "Production Field 1 (Argo North-Home Minor)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-05/Row 5/DS Splits",
            },
        },
        "FF2": {
            "7-08": {
                "Row8": "Foundation Field 2 (Dennis Zuber)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-08/Row 8, 9/Pass A/DS Splits",
                "Row9": "Foundation Field 2 (Dennis Zuber)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-08/Row 8, 9/Pass B/DS Splits",
                "Row10": "Foundation Field 2 (Dennis Zuber)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-08/Row 7, 10/Pass B/DS Splits",
                "Row11": "Foundation Field 2 (Dennis Zuber)/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/7-08/Row 6, 11/Pass B/DS Splits",
            }
        },
        "FC22": {
            "StripTrial": {
                "6-21": {
                    "Row1b": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 1b, 6a/Pass A/DS Splits",
                    "Row2b": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 2b, 5a/Pass A/DS Splits",
                    "Row3b": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 3b, 4a/Pass A/DS Splits",
                    "Row4b": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 4b, 3a/Pass A/DS Splits",
                    "Row5b": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 5b, 2a/Pass A/DS Splits",
                    "Row6b": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 6b, 1a/Pass A/DS Splits",
                    "Row6a": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 1b, 6a/Pass B/DS Splits",
                    "Row5a": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 2b, 5a/Pass B/DS Splits",
                    "Row4a": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 3b, 4a/Pass B/DS Splits",
                    "Row3a": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 4b, 3a/Pass B/DS Splits",
                    "Row2a": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 5b, 2a/Pass B/DS Splits",
                    "Row1a": "Farmer City 2022/Strip Trial/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-21/Row 6b, 1a/Pass B/DS Splits",
                },
            },
            "SmallPlot": {
                "Planting1": {
                    "6-16": {
                        "Row5": "Farmer City 2022/Small Plot/Planting 1/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/6-16/Row 8, 5/Pass B/Rel Plots/",
                    },
                },
            },
        },
    }
