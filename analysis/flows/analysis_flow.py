"""A test flow for using metaflow for analysis."""
from metaflow import FlowSpec, Parameter, batch, step

from analysis.scripts.analyze_data import analyze_data
from analysis.scripts.pull_inferenced_data import pull_inferenced_data
from analysis.scripts.run_analysis import run_analysis
from analysis.utils.analysis_params import AnalysisParameters


class AnalysisFlow(FlowSpec):
    """Need to think about what parameters will be needed to run this from beginning to end."""

    # parameters
    job_name = Parameter("job_name", help="name of job", required=True)
    field = Parameter("field", help="field to process: [PF1, FF1, FF2, FC22,]")
    small_plot_or_strip_trial = Parameter(
        "sp_or_st", help="small plot or strip trial (only for FC22)", default=None
    )
    date = Parameter("date", help="date of data")
    row = Parameter("row", help="row to process")
    unstaked = Parameter(
        "unstaked", help="indicates if the row is unstaked", default=False
    )

    # maybe just give the path ^ that doesn't seem necessary and path should be easily obtainable from the database

    # steps
    @step
    def start(self):
        """Start of flow: initialization."""
        # file naming convention: abbreviatedField_m-dd_RowX_s3_keys.json for staked rows
        # i.e. PF1_6-27_Row2_s3_keys.json
        base_name = f"{self.field}_{self.date}_{self.row}"
        if self.small_plot_or_strip_trial is not None:
            sp_or_st = "SP" if "small" in self.small_plot_or_strip_trial else "ST"
            base_name = f"{self.field}_{sp_or_st}_{self.date}_{self.row}"
        if self.unstaked:
            base_name = f"{base_name}_unstaked"

        self.base_name = base_name
        self.json_file_name = f"{base_name}_s3_keys.json"
        self.next(self.pull_data)

    @step
    def pull_data(self):
        """Pull the data from s3 using a s3_keys.json file."""
        pull_inferenced_data(
            input=f"{self.base_name}_s3_keys.json",
            local_path=AnalysisParameters.LOCAL_PATH_ROOT,
        )
        self.next(self.eval_neighboring_row_thresholds)

    @step
    def eval_neighboring_row_thresholds(self):
        """Thresholds out the neighboring rows."""
        local_directory = f"{AnalysisParameters.LOCAL_PATH_ROOT}/FF2_7-08_Row8/"
        run_analysis(dir=local_directory)
        analyze_data(dataset=f"{local_directory}/measurements_df.gz", threshold=True)
        # analyze_data.py: run with --threshold
        self.next(self.analyze_data)

    @batch(
        image="475283710372.dkr.ecr.us-east-1.amazonaws.com",
        queue="rogues-processing",
        cpu=128,
        memory=256000,
    )
    @step
    def analyze_data(self):
        """Run the analysis."""
        # analyze_data.py: run with -m 'build'
        # analyze_data.py: run with model produced from previous command, turn on all plotting
        self.next(self.end)

    @step
    def end(self):
        """End of flow."""
        print("Analysis flow complete")


if __name__ == "__main__":
    AnalysisFlow()
