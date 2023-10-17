"""Utilities for mapping keys from one MirrorPath to another."""

from typing import List

from S3MP.keys import KeySegment
from S3MP.mirror_path import MirrorPath

from ground_data_processing.utils.absolute_segment_groups import ProdInferenceSegments
from ground_data_processing.utils.relative_segment_groups import (
    DataTypeSegment,
    DateAndRowSegments,
    DSSplitSegments,
    FCPlotDesignationSegments,
    FieldNameSegment,
    ModelInferenceSegments,
    PlantingTrialRootSegments,
    PlotSplitVideoSegments,
    ResolutionSegment,
    RowSplitVideoSegments,
    SegmentGroupMetaclass,
)
from ground_data_processing.utils.s3_constants import (
    DataTypes,
    InferenceMethods,
    ModelNames,
)


class Keymap:
    """S3 Keymap."""

    def __init__(
        self,
        input_prefix_segment_groups: List[SegmentGroupMetaclass],
        preserve_segment_groups: List[SegmentGroupMetaclass],
        output_prefix_segment_groups: List[SegmentGroupMetaclass],
        output_insert_segments: List[KeySegment],
    ):
        """
        Take an s3 key and map it to a new key.

        Process is as follows:
        Input key: <input_prefix_segment_group>/<preserve_segment_group>/<...>
        Output key: <output_prefix_segment_group>/<output_insert_segments>/<preserve_segment_group>

        :param input_prefix_segment_groups: Segment groups prior to the preserve segment group.
        :param preserve_segment_group: Segment group to preserve.
        :param output_prefix_segment_groups: Segment groups in the output key place before the preserve segment group.
        :param output_insert_segments: Segments to insert between the output prefix and preserve segment groups.
        """
        # Handle single by casting to List
        if not isinstance(input_prefix_segment_groups, list):
            input_prefix_segment_groups = [input_prefix_segment_groups]
        if not isinstance(preserve_segment_groups, list):
            preserve_segment_groups = [preserve_segment_groups]
        if not isinstance(output_prefix_segment_groups, list):
            output_prefix_segment_groups = [output_prefix_segment_groups]
        if not isinstance(output_insert_segments, list):
            output_insert_segments = [output_insert_segments]

        self.input_prefix_segment_groups = input_prefix_segment_groups
        self.preserve_segment_groups = preserve_segment_groups
        self.output_prefix_segment_groups = output_prefix_segment_groups
        self.output_insert_segments = output_insert_segments

    def get_preserve_segments(self, input_mp: MirrorPath) -> List[KeySegment]:
        """Get the segments to preserve from a MirrorPath."""
        input_prefix_depth = sum(
            len(seg_group.unique_folder_depths())
            for seg_group in self.input_prefix_segment_groups
        )
        preserve_segment_idxs = []
        offset = 0
        for seg_group in self.preserve_segment_groups:
            preserve_segment_idxs.extend(
                [idx + offset for idx in seg_group.unique_folder_depths()]
            )
            offset = preserve_segment_idxs[-1] + 1

        return [
            input_mp.get_key_segment(idx + input_prefix_depth)
            for idx in preserve_segment_idxs
        ]

    def get_preserve_strs(self, input_mp: MirrorPath) -> List[str]:
        """Get the preserve strings from an input MirrorPath."""
        preserve_segments = self.get_preserve_segments(input_mp)
        return [seg.name for seg in preserve_segments]

    def apply(self, input_mp: MirrorPath) -> MirrorPath:
        """Apply the keymap to a MirrorPath."""
        preserve_segments = self.get_preserve_segments(input_mp)
        output_prefix_depth = sum(
            len(seg_group.unique_folder_depths())
            for seg_group in self.output_prefix_segment_groups
        )
        output_mp = input_mp.trim(output_prefix_depth)
        output_mp.key_segments.extend(self.output_insert_segments)
        output_mp.key_segments.extend(preserve_segments)
        return output_mp


PROD_INFERENCE_STATS_KEYMAP = Keymap(
    input_prefix_segment_groups=[
        FieldNameSegment,
        DataTypeSegment,
        ModelInferenceSegments,
    ],
    preserve_segment_groups=[DateAndRowSegments, DSSplitSegments],
    output_prefix_segment_groups=[
        FieldNameSegment,
    ],
    output_insert_segments=[
        DataTypeSegment.DATA_TYPE(DataTypes.IMAGES),
    ],
)

PROD_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP = Keymap(
    input_prefix_segment_groups=[
        FieldNameSegment,
        DataTypeSegment,
    ],
    preserve_segment_groups=[DateAndRowSegments, DSSplitSegments],
    output_prefix_segment_groups=[
        FieldNameSegment,
    ],
    output_insert_segments=[
        DataTypeSegment.DATA_TYPE(DataTypes.UNFILTERED_MODEL_INFERENCE),
        ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        ProdInferenceSegments.PREPROC_METHOD(InferenceMethods.SQUARE_CROP),
    ],
)
PROD_SPLIT_ROW_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP = Keymap(
    input_prefix_segment_groups=[
        FieldNameSegment,
        DataTypeSegment,
    ],
    preserve_segment_groups=[
        DateAndRowSegments,
        RowSplitVideoSegments,
        DSSplitSegments,
    ],
    output_prefix_segment_groups=[
        FieldNameSegment,
    ],
    output_insert_segments=[
        DataTypeSegment.DATA_TYPE(DataTypes.UNFILTERED_MODEL_INFERENCE),
        ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        ProdInferenceSegments.PREPROC_METHOD(InferenceMethods.SQUARE_CROP),
    ],
)

PLOT_TRIAL_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP = Keymap(
    input_prefix_segment_groups=[
        PlantingTrialRootSegments,
        DataTypeSegment,
    ],
    preserve_segment_groups=[
        DateAndRowSegments,
        ResolutionSegment,
        RowSplitVideoSegments,
        PlotSplitVideoSegments,
    ],
    output_prefix_segment_groups=[
        PlantingTrialRootSegments,
    ],
    output_insert_segments=[
        DataTypeSegment.DATA_TYPE(DataTypes.UNFILTERED_MODEL_INFERENCE),
        ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        ProdInferenceSegments.PREPROC_METHOD(InferenceMethods.SQUARE_CROP),
    ],
)


FC_INFERENCE_STATS_KEYMAP = Keymap(
    input_prefix_segment_groups=[
        PlantingTrialRootSegments,
        DataTypeSegment,
        ModelInferenceSegments,
    ],
    preserve_segment_groups=[FCPlotDesignationSegments, ResolutionSegment],
    output_prefix_segment_groups=[
        FCPlotDesignationSegments,
    ],
    output_insert_segments=[
        DataTypeSegment.DATA_TYPE(DataTypes.IMAGES),
    ],
)
