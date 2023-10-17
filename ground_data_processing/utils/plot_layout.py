"""Plot layout logic."""
from S3MP.mirror_path import MirrorPath

plot_layout = [
    ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],  # this is plot 1
    ["1HL", "2HL", "D3", "2HL", "2HL", "Mix", "1HL", "1HL", "2HL", "1HL", "2HL", "1HL"],
    ["1HL", "D3", "1HL", "Mix", "1HL", "D3", "1HL", "2HL", "2HL", "1HL", "1HL", "Mix"],
    [
        "Mix",
        "1HL",
        "1HL",
        "1HL",
        "2HL",
        "Mix",
        "1HL",
        "1HL",
        "1HL",
        "1HL",
        "1HL",
        "2HL",
    ],
    [
        "1HL",
        "1HL",
        "2HL",
        "2HL",
        "Mix",
        "2HL",
        "2HL",
        "2HL",
        "2HL",
        "2HL",
        "Mix",
        "2HL",
    ],
    [
        "Mix",
        "1HL",
        "2HL",
        "Mix",
        "1HL",
        "Mix",
        "2HL",
        "2HL",
        "1HL",
        "1HL",
        "2HL",
        "2HL",
    ],
    [
        "1HL",
        "2HL",
        "Mix",
        "2HL",
        "2HL",
        "1HL",
        "2HL",
        "1HL",
        "2HL",
        "2HL",
        "Mix",
        "1HL",
    ],
    [
        "2HL",
        "2HL",
        "2HL",
        "2HL",
        "1HL",
        "1HL",
        "1HL",
        "2HL",
        "1HL",
        "1HL",
        "2HL",
        "1HL",
    ],
    ["1HL", "Mix", "Mix", "1HL", "1HL", "2HL", "2HL", "HH", "1HL", "2HL", "1HL", "1HL"],
    ["2HL", "2HL", "2HL", "1HL", "D3", "1HL", "1HL", "1HL", "2HL", "2HL", "2HL", "D3"],
    ["D3", "2HL", "1HL", "Mix", "Mix", "1HL", "HH", "1HL", "2HL", "1HL", "Mix", "1HL"],
    ["2HL", "1HL", "2HL", "1HL", "2HL", "HH", "2HL", "1HL", "2HL", "1HL", "2HL", "2HL"],
    ["2HL", "1HL", "1HL", "1HL", "1HL", "D3", "1HL", "1HL", "1HL", "Mix", "1HL", "D3"],
    ["HH", "2HL", "1HL", "2HL", "HH", "HH", "2HL", "1HL", "1HL", "2HL", "2HL", "1HL"],
    ["1HL", "2HL", "1HL", "2HL", "2HL", "2HL", "D3", "HH", "1HL", "1HL", "2HL", "1HL"],
    ["1HL", "2HL", "1HL", "HH", "1HL", "1HL", "Mix", "D3", "2HL", "2HL", "2HL", "Mix"],
    ["1HL", "Mix", "1HL", "2HL", "Mix", "2HL", "1HL", "1HL", "HH", "2HL", "1HL", "2HL"],
    ["2HL", "1HL", "1HL", "2HL", "1HL", "1HL", "HH", "1HL", "2HL", "1HL", "1HL", "1HL"],
    ["D3", "1HL", "1HL", "2HL", "1HL", "1HL", "D3", "2HL", "1HL", "1HL", "2HL", "2HL"],
    ["1HL", "1HL", "1HL", "1HL", "2HL", "1HL", "1HL", "1HL", "Mix", "HH", "1HL", "1HL"],
    ["1HL", "2HL", "HH", "2HL", "2HL", "2HL", "2HL", "Mix", "1HL", "1HL", "1HL", "1HL"],
    ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],  # this is plot 22
]
base_types = ["F1", "F2", "F3", "M1"] * 3


def get_reverse_flag_from_pass_mp(pass_mp: MirrorPath):
    """Get the reverse flag from the pass MP."""
    return pass_mp.key_segments[-1].name == "Pass B"


def get_row_number_from_pass_mp(pass_mp: MirrorPath):
    """Get the row number from the pass MP."""
    row_numbers = pass_mp.key_segments[-3].name.split(", ")
    return (
        int(row_numbers[1])
        if get_reverse_flag_from_pass_mp(pass_mp)
        else int(row_numbers[0])
    )


def convert_plot_idx_to_plot_number(plot_idx: int, pass_reverse_flag: bool):
    """Convert the plot index to the plot number."""
    return 21 - plot_idx if pass_reverse_flag else plot_idx + 1


def get_rogue_and_base_type(row_number, plot_number):
    """Get rogue and base type."""
    return plot_layout[plot_number - 1][row_number - 1], base_types[row_number - 1]


"""
Nicely formatted with tabs:


 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B
1HL	2HL	D3 	2HL	2HL	Mix	1HL	1HL	2HL	1HL	2HL	1HL
1HL	D3  1HL	Mix	1HL	D3 	1HL	2HL	2HL	1HL	1HL	Mix
Mix	1HL	1HL	1HL	2HL	Mix	1HL	1HL	1HL	1HL	1HL	2HL
1HL	1HL	2HL	2HL	Mix	2HL	2HL	2HL	2HL	2HL	Mix	2HL
Mix	1HL	2HL	Mix	1HL	Mix	2HL	2HL	1HL	1HL	2HL	2HL
1HL	2HL	Mix	2HL	2HL	1HL	2HL	1HL	2HL	2HL	Mix	1HL
2HL	2HL	2HL	2HL	1HL	1HL	1HL	2HL	1HL	1HL	2HL	1HL
1HL	Mix	Mix	1HL	1HL	2HL	2HL	HH 	1HL	2HL	1HL	1HL
2HL	2HL	2HL	1HL	D3 	1HL	1HL	1HL	2HL	2HL	2HL	D3
D3 	2HL	1HL	Mix	Mix	1HL	HH 	1HL	2HL	1HL	Mix	1HL
2HL	1HL	2HL	1HL	2HL	HH 	2HL	1HL	2HL	1HL	2HL	2HL
2HL	1HL	1HL	1HL	1HL	D3 	1HL	1HL	1HL	Mix	1HL	D3
HH 	2HL	1HL	2HL	HH 	HH 	2HL	1HL	1HL	2HL	2HL	1HL
1HL	2HL	1HL	2HL	2HL	2HL	D3 	HH 	1HL	1HL	2HL	1HL
1HL	2HL	1HL	HH 	1HL	1HL	Mix	D3 	2HL	2HL	2HL	Mix
1HL	Mix	1HL	2HL	Mix	2HL	1HL	1HL	HH 	2HL	1HL	2HL
2HL	1HL	1HL	2HL	1HL	1HL	HH 	1HL	2HL	1HL	1HL	1HL
D3 	1HL	1HL	2HL	1HL	1HL	D3 	2HL	1HL	1HL	2HL	2HL
1HL	1HL	1HL	1HL	2HL	1HL	1HL	1HL	Mix	HH 	1HL	1HL
1HL	2HL	HH 	2HL	2HL	2HL	2HL	Mix	1HL	1HL	1HL	1HL
 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B 	 B
"""
