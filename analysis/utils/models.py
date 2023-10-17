"""Model Class(es) for determining rogues from measurements."""
import json
import os

# import matplotlib
import mlflow
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import analysis.utils.analysis_tools as tools
import analysis.utils.plotting_tools as plot_tools
from analysis.utils.analysis_params import AnalysisParameters, ModelParameters
from analysis.utils.peak_detector import PeakDetector

# matplotlib.use("tkAgg")


class Model:
    """Builds a stat model from provided data."""

    def __init__(
        self,
        name: str = None,
        data: pd.DataFrame() = None,
        model: dict | str = None,
        unstaked: bool = False,
        plot_path: str = None,
        plot_features: bool = False,
        plot_splits: bool = False,
        public: bool = False,
        log: bool = True,
    ) -> None:
        """Init."""
        # check for valid inputs
        if name is None:
            raise RuntimeError("name is None, please provide a valid name")
        if data is None:
            raise RuntimeError("data is None, please provide a valid dataset")
        if plot_features or plot_splits:
            if plot_path is None:
                raise RuntimeError(
                    "plot_path is None, please provide a valid plot_path"
                )

        self._kBuild = (
            True if model is None else False
        )  # indicates if the model was built or loaded
        self.name = self.get_name(name)
        self.data = data
        self.unstaked = unstaked
        self.plot_path = plot_path
        self.plot_features = plot_features
        self.plot_splits = plot_splits
        self.public = public
        self.log = log
        self.oblique_weight = 1
        self.window = 30
        if log:
            mlflow.set_tracking_uri(ModelParameters.MLFLOW_URI)
            mlflow.set_experiment(ModelParameters.MLFLOW_EXPERIMENT_NAME)
            mlflow.start_run(run_name=self.name)
        self.features = (
            AnalysisParameters.ANALYSIS_FEATURES
            if "Stats-length" in self.data.columns
            else AnalysisParameters.ANALYSIS_FEATURES_OBLIQUE
        )
        if "Stats-length" in self.data.columns:
            self.features = self.features + ["Stats-rectangular"]
        else:
            self.features = self.features + [
                "Stats-rectangular-nadir",
                "Stats-rectangular-oblique",
            ]

        # extract metrics for easier access
        print("[INFO] extracting metrics")
        self.extract_metrics()

        # extract rogue locations
        print("[INFO] extracting rogue locations")
        self.extract_rogues()

        # initialize model
        print("[INFO] initializing model")
        # if model is dict
        if isinstance(model, dict):
            self.model_params = model
        elif model == "build":
            self.initialize_static_model()
        elif model == "dynamic":
            self.initialize_dynamic_model()
        else:
            raise RuntimeError(
                f"[ERROR] model {model} is not an acceptable model type, please provide a valid model"
            )

    def __del__(self):
        """Destructor to close mlflow run."""
        if mlflow.active_run():
            mlflow.end_run()

    def get_name(self, name):
        """Get the name of the model."""
        if "/" in name:
            name = name.split("/")[-1]
        return name

    def make_threshold_plot(self):
        """Make the threshold plot."""
        if "Centroids" in self.data.columns:
            plot_tools.plot_centroids(self.data["Centroids"], self.plot_path)
        else:
            plot_tools.plot_centroids(self.data["Centroids-nadir"], self.plot_path)
            plot_tools.plot_centroids(self.data["Centroids-oblique"], self.plot_path)

    def extract_rogues(self):
        """
        Extract rogue locations from the data.

        The possible rogue values are:
        Hybrid Rogue (high confidence)
        Hybrid Low Rogue (low confidence)
        Delay Rogue
        Other
        Normal
        """
        rogue_values = ["Hybrid Rogue", "Hybrid Low Rogue", "Delay Rogue", "Other"]
        self.rogues = self.data[
            self.data[AnalysisParameters.ROGUE].str.contains("|".join(rogue_values))
        ]
        print(np.unique(self.data[AnalysisParameters.ROGUE].values))
        for x in np.unique(self.data[AnalysisParameters.ROGUE].values):
            if x == "???":
                x = "\\?\\?\\?"
            print(x, sum(self.data[AnalysisParameters.ROGUE].str.contains(x)))

        # we are only liable for high confidence rogues
        self.rogue_indices = self.rogues.index[
            self.rogues[AnalysisParameters.ROGUE].str.contains("Hybrid Rogue")
        ]
        self.all_rogue_indices = self.rogues.index[
            ~self.rogues[AnalysisParameters.ROGUE].str.contains("Normal")
        ]
        self.other_rogues = self.data.loc[
            self.all_rogue_indices.drop(self.rogue_indices)
        ].index

        # drop any rogues with index less than 30
        self.rogue_indices = self.rogue_indices[self.rogue_indices > 30]
        if len(self.rogue_indices) == 0:
            self._hasRogues = False
        else:
            self._hasRogues = True
        self.all_rogue_indices = self.all_rogue_indices[self.all_rogue_indices > 30]
        self.other_rogues = self.other_rogues[self.other_rogues > 30]

    def add_rectangular_features(self):
        """
        Add rectangular features to the self.data DataFrame.

        The rectangular features are:
        Stats('rectangular', length * width)
        """
        rect_stats = []
        if "Stats-length" not in self.data.columns:
            rect_stats = [
                tools.Stats(
                    "rectangular-nadir",
                    np.multiply(
                        self.data.loc[r, "Stats-length-nadir"].x,
                        self.data.loc[r, "Stats-width-nadir"].x,
                    ),
                )
                for r in self.data.index
            ]
            self.data["Stats-rectangular-nadir"] = rect_stats
            rect_stats = [
                tools.Stats(
                    "rectangular-oblique",
                    np.multiply(
                        self.data.loc[r, "Stats-length-oblique"].x,
                        self.data.loc[r, "Stats-width-oblique"].x,
                    ),
                )
                for r in self.data.index
            ]
            self.data["Stats-rectangular-oblique"] = rect_stats
        else:
            rect_stats = [
                tools.Stats(
                    "rectangular",
                    np.multiply(
                        self.data.loc[r, "Stats-length"].x,
                        self.data.loc[r, "Stats-width"].x,
                    ),
                )
                for r in self.data.index
            ]
            self.data["Stats-rectangular"] = rect_stats

    def extract_metrics(self):
        """Extract metrics from the data."""
        self.add_rectangular_features()

        # extract stats
        stat_features = [x for x in self.features if "Stats" in x]
        for feat in stat_features:
            stat_dict = [x.to_dict() for x in self.data[feat]]
            for stat in AnalysisParameters.STATS_KEYS[
                1:
            ]:  # params.STATS_KEYS[0] is 'name', don't need to plot name
                self.data[f"{feat}_{stat}"] = [x[stat] for x in stat_dict]

        # extract total number of masked pixels usings 'Stats-area-nadir'
        self.data["Stats-total-masked-nadir"] = [
            sum(x.x) / (1024**2) for x in self.data["Stats-area-nadir"]
        ]

        # extract metrics from cumulative distribution of the y values of the masks.
        d = []
        for x in self.data["cumulative-distribution-mask-y-values"].values:
            d.append(np.asarray(x))
        # d = np.asarray(d)
        self.data["Stats-cum-dist-oblique_mean"] = np.array(
            [np.mean(x) if len(x) > 0 else 0 for x in d]
        )
        self.data["Stats-cum-dist-oblique_d10"] = np.array(
            [1024 - np.where(x < 0.1)[0][-1] if len(x) > 0 else 0 for x in d]
        )
        self.data["Stats-cum-dist-oblique_d30"] = np.array(
            [1024 - np.where(x < 0.3)[0][-1] if len(x) > 0 else 0 for x in d]
        )
        self.data["Stats-cum-dist-oblique_d50"] = np.array(
            [1024 - np.where(x < 0.5)[0][-1] if len(x) > 0 else 0 for x in d]
        )
        self.data["Stats-cum-dist-oblique_d70"] = np.array(
            [1024 - np.where(x < 0.7)[0][-1] if len(x) > 0 else 0 for x in d]
        )
        ModelParameters.MODEL_PARAMS = ModelParameters.MODEL_PARAMS + [
            "Stats-total-masked-nadir",
            "Stats-cum-dist-oblique_mean",
            "Stats-cum-dist-oblique_d10",
            "Stats-cum-dist-oblique_d30",
            "Stats-cum-dist-oblique_d50",
            "Stats-cum-dist-oblique_d70",
        ]

    def initialize_static_model(self):
        """Initialize the model parameters."""
        self.model_params = {}

        ModelParameters.MODEL_PARAMS = ModelParameters.MODEL_PARAMS + [
            f"{x}_{y}"
            for x in self.features
            for y in AnalysisParameters.STATS_KEYS
            if "Stats" in x and "name" not in y
        ]
        for x in ModelParameters.MODEL_PARAMS:
            if x in ["cut", "min_viable_cut"]:
                self.model_params[x] = 0
            elif x == "window":
                self.model_params[x] = 30
            elif "Stats" in x and "median" not in x:
                self.model_params[x] = {
                    ModelParameters.SIGMA: 2,
                    # sometimes the influence is negative, so we need to take the absolute value
                    ModelParameters.INFLUENCE: np.abs(
                        np.mean(self.data[x].values) / np.max(self.data[x].values) / 3
                    ),
                    ModelParameters.ACTIVE: True,
                }
            else:
                self.model_params[x] = {
                    ModelParameters.SIGMA: 2,
                    ModelParameters.INFLUENCE: 0.0,
                    ModelParameters.ACTIVE: False,
                }

        return self.model_params

    def get_num_true_false(self, signal) -> tuple[int, int]:
        """
        Take in a signal and return the number of true positives and false positives.

        Note: neighbors of rogues are not counted as false positives or true positives.
        Note: signal is a list of 0s and 1s, where 1 indicates a rogue.

        Args:
        signal (np.array | list): signal

        Returns:
        num_true (int): number of true positives
        num_false (int): number of false positives
        neighbors (list): list of true false indicating whether plant is
        neighbor to a high confidence rogue
        """
        num_true, num_false = -1, -1
        signal = np.array(signal)
        if len(signal) < 1:
            return num_true, num_false

        num_true = len(np.where(signal[self.rogue_indices.values] > 0)[0])
        missed_rogue_indices = self.rogue_indices[
            signal[self.rogue_indices.values] == 0
        ]
        neighbors = np.full(signal.shape, False)
        for x in self.rogue_indices:
            # look for neighbors with positive signal
            rogue_idx = x
            _mark_if_found = False
            if rogue_idx in missed_rogue_indices:
                _mark_if_found = True
            # backwards
            while x > 0 and signal[x - 1] == 1:
                if _mark_if_found:
                    num_true += 1
                    _mark_if_found = False
                    missed_rogue_indices = missed_rogue_indices.drop(rogue_idx)
                neighbors[x - 1] = True
                x -= 1
            x = rogue_idx
            # forwards
            while x < len(signal) - 1 and signal[x + 1] == 1:
                neighbors[x + 1] = True
                if _mark_if_found:
                    num_true += 1
                    _mark_if_found = False
                    missed_rogue_indices = missed_rogue_indices.drop(rogue_idx)
                x += 1

        num_false = len(np.where(signal > 0)[0]) - num_true - sum(neighbors)

        return num_true, num_false, neighbors

    def build_static_model(self, no_oblique=False, no_nadir=False):
        """
        Build the model.

        The basic outline is:
        1. build default model
        2. tune influence for each metric
        3. tune sigma for each metric
        4. deactivate metrics that are not useful
        5. tune cut (TODO)
        """
        if no_oblique:
            for x in self.model_params:
                if "oblique" in x:
                    self.model_params[x][ModelParameters.ACTIVE] = False

        # tune influence, identify a reasonable cut
        rogue_signals = []
        signals = np.zeros(len(self.data))
        print("[INFO] tuning influence...identify a reasonable cut")
        max_string_len = max([len(x) for x in self.model_params])
        for x in self.model_params:
            if "Stats" not in x:
                continue
            if not self.model_params[x][ModelParameters.ACTIVE]:
                continue
            if np.isnan(self.model_params[x][ModelParameters.INFLUENCE]):
                self.model_params[x][ModelParameters.ACTIVE] = False
                continue
            # now tune influence
            last = 0
            done = False
            # incrememnt = 0.001
            while not done:
                last = self.model_params[x][ModelParameters.INFLUENCE]
                # a little trick to get the next power of 10
                increment = (
                    0.001 if last == 0 else 10 ** (int(round(np.log10(last / 10), 6)))
                )
                self.model_params[x][ModelParameters.INFLUENCE] += increment
                (
                    influenced_signal,
                    influenced_mean,
                    _,
                ) = tools.get_influenced_signal_with_mean(
                    self.data[x].values,
                    self.model_params["window"],
                    self.model_params[x][ModelParameters.SIGMA],
                    self.model_params[x][ModelParameters.INFLUENCE],
                )
                if "oblique" in x:
                    influenced_signal *= self.oblique_weight
                peaks_data, _ = find_peaks(
                    self.data[x].values[30:],
                    prominence=2,
                )
                peaks_mean, _ = find_peaks(
                    influenced_mean[30:],
                    prominence=2,
                )
                num_mean_peaks_in_data = sum([x in peaks_data for x in peaks_mean])
                if (
                    num_mean_peaks_in_data
                    >= ModelParameters.CORRELATION_THRESHOLD * len(peaks_data)
                ):
                    done = True
                if self.model_params[x][ModelParameters.INFLUENCE] > 1:
                    # if you get to 1, you're not doing anything useful.
                    done = True
                    self.model_params[x][ModelParameters.ACTIVE] = False

            self.model_params[x][ModelParameters.INFLUENCE] = round(last, 4)
            print(f"{x:>{max_string_len}}, {self.model_params[x]}")
            (
                influenced_signal,
                inf_mean,
                num_std_devs,
            ) = tools.get_influenced_signal_with_mean(
                self.data[x].values,
                self.model_params["window"],
                self.model_params[x][ModelParameters.SIGMA],
                self.model_params[x][ModelParameters.INFLUENCE],
            )
            if "oblique" in x:
                influenced_signal *= self.oblique_weight
            signals = np.add(signals, influenced_signal)
            rogue_signals.append(influenced_signal[self.rogue_indices])

        rogue_signals = np.array(rogue_signals)
        # rogue_signal_totals = signals[self.rogue_indices]
        self.data["signal"] = signals
        self.rogues = self.data.loc[self.all_rogue_indices, :]
        print("initial signals:")
        print(
            self.rogues.loc[
                self.rogue_indices, ["dataset", "name", "rogue", "signal"]
            ].to_string()
        )

        active_params = []
        for x in self.model_params:
            if "Stats" not in x:
                continue
            if self.model_params[x][ModelParameters.ACTIVE]:
                active_params.append(x)

        while active_params:
            # tune sigma of active params
            p = active_params.pop(0)

            # set baseline signal
            influenced_signal, _, num_std_devs = tools.get_influenced_signal_with_mean(
                self.data[p].values,
                self.model_params["window"],
                self.model_params[p][ModelParameters.SIGMA],
                self.model_params[p][ModelParameters.INFLUENCE],
            )

            rogue_signal_base = influenced_signal[self.rogue_indices]
            rogue_signal = rogue_signal_base[:]

            # tune sigma, exit loop when sigma is tight enough to turn off rogue signal
            # or when you hit 5 sigma (extraordinary evidence for extraordinary claims)
            last = 0
            while all(x == y for x, y in zip(rogue_signal, rogue_signal_base)):
                if round(self.model_params[p][ModelParameters.SIGMA], 1) >= 5.0:
                    # exit loop if you hit 5 sigma
                    break
                last = self.model_params[p][ModelParameters.SIGMA]
                self.model_params[p][ModelParameters.SIGMA] += 0.01
                influenced_signal, _, _ = tools.get_influenced_signal_with_mean(
                    self.data[p].values,
                    self.model_params["window"],
                    self.model_params[p][ModelParameters.SIGMA],
                    self.model_params[p][ModelParameters.INFLUENCE],
                )
                rogue_signal = influenced_signal[self.rogue_indices]
            self.model_params[p][ModelParameters.SIGMA] = round(last, 4)
            print(f"{p:>{max_string_len}}, {self.model_params[p]}")

        # determine core metrics for second threshold
        metrics_with_purity = []
        signals = np.zeros(len(self.data))
        for x in self.model_params:
            if "Stats" not in x:
                continue
            if not self.model_params[x][ModelParameters.ACTIVE]:
                continue
            influenced_signal, _, _ = tools.get_influenced_signal_with_mean(
                self.data[x].values,
                self.model_params["window"],
                self.model_params[x][ModelParameters.SIGMA],
                self.model_params[x][ModelParameters.INFLUENCE],
            )
            signals = np.add(signals, influenced_signal)
            num_true, num_false, _ = self.get_num_true_false(influenced_signal)
            metrics_with_purity.append((x, num_true, num_false, influenced_signal))

        metrics_with_purity = sorted(
            metrics_with_purity, key=lambda x: (x[1], -x[2]), reverse=True
        )
        print("~" * 80)
        for x in metrics_with_purity:
            print(f"{x[0]:>{max_string_len}}: {x[1]:>5} true, {x[2]:>5} false")

        purity = []
        signal_length = len(metrics_with_purity[0][3])
        for i in range(len(metrics_with_purity)):
            s = np.zeros(signal_length)
            for x in [x[3] for x in metrics_with_purity[: i + 1]]:
                s = np.add(s, x)
            cut = min(s[self.rogue_indices]) - 1
            num_true, num_false, _ = self.get_num_true_false(s > cut)
            purity.append((num_true, num_false, i))

        purity.sort(key=lambda x: (x[0], -x[1], x[2]), reverse=True)

        index = purity[0][2] if purity[1][1] - purity[0][1] > 10 else purity[1][2]
        most_num_true = max([x[1] for x in metrics_with_purity])
        print("~" * 80)
        print("min viable model:")
        print(index, len(purity))
        if index < len(metrics_with_purity) // 2:
            for i in range(index + 1):
                # activate metrics that contribute to min_signal
                print(
                    metrics_with_purity[i][0],
                    metrics_with_purity[i][1],
                    metrics_with_purity[i][2],
                )
                self.model_params[metrics_with_purity[i][0]][
                    ModelParameters.MIN_VIABLE
                ] = True
        print("~" * 80)

        # drop/remove/deactivate metrics that don't contribute to min_signal
        print("dropping metrics that don't contribute to min_signal")
        print(" most num true:", most_num_true)
        for x in self.model_params:
            if "Stats" not in x:
                continue
            if not self.model_params[x][ModelParameters.ACTIVE]:
                continue
            if ModelParameters.MIN_VIABLE in self.model_params[x]:
                continue  # don't deactivate min_viable metrics
            # if sum(rogue_signals[i])/len(self.rogue_indices) <= 0.8:
            try:
                metric_index = [
                    i for i, z in enumerate(metrics_with_purity) if z[0] == x
                ][0]
            except Exception:
                print(f"[ERROR] {x} not found in metrics_with_purity")
                continue
            print(
                f"{x:>{max_string_len}} {round(metrics_with_purity[metric_index][1]/len(self.rogue_indices), 4):<6} {metrics_with_purity[metric_index][3][self.rogue_indices]}"
            )
            if (
                metrics_with_purity[metric_index][1] / len(self.rogue_indices) < 0.5
            ):  # metric misses more than 50% of rogues
                self.model_params[x][ModelParameters.ACTIVE] = False
                print(f"deactivating {x}")
                continue
            print(
                f"{x:>{max_string_len}} {round(metrics_with_purity[metric_index][1]/len(self.rogue_indices), 4):<6} {metrics_with_purity[metric_index][3][self.rogue_indices]}"
            )

        self.model_params["cut"] = min(signals[self.rogue_indices]) - 1
        self.model_params["min_viable_cut"] = index

        print("cut", self.model_params["cut"])
        print("min_viable_cut", self.model_params["min_viable_cut"])

        self.save_model()

    def evaluate_static_model(self):
        """Evaluate the accuracy of the model."""
        rogue_signals = np.zeros(len(self.data))
        min_viable_signals = np.zeros(len(self.data))
        rogue_signals_by_purity = np.zeros(len(self.data))
        pure_signals = []
        metric_with_purity = []
        print("cut", self.model_params["cut"])
        print("min_viable_cut", self.model_params["min_viable_cut"])
        print("Active params are:")
        for x in self.model_params:
            if "Stats" not in x:
                continue
            if not self.model_params[x][ModelParameters.ACTIVE]:
                continue
            self.data = self.data.copy()  # work around for PerformanceWarning
            (
                influenced_signal,
                self.data[f"{x}_inf_mean"],
                self.data[f"{x}_num_std"],
            ) = tools.get_influenced_signal_with_mean(
                self.data[x].values,
                self.model_params["window"],
                self.model_params[x][ModelParameters.SIGMA],
                self.model_params[x][ModelParameters.INFLUENCE],
            )
            num_true, num_false, _ = self.get_num_true_false(influenced_signal)
            if num_false != 0:
                metric_with_purity.append((x, float(num_true) / float(num_false)))
                if float(num_true) / float(num_false) > 0.09:  # 2.4
                    # if float(num_true)/float(num_false) > 0.1:
                    pure_signals.append(x)
                    rogue_signals_by_purity = np.add(
                        rogue_signals_by_purity, influenced_signal
                    )
            else:
                metric_with_purity.append((x, 100))
                pure_signals.append(x)
                rogue_signals_by_purity = np.add(
                    rogue_signals_by_purity, influenced_signal
                )
            if "oblique" in x:
                influenced_signal *= self.oblique_weight

            if ModelParameters.MIN_VIABLE in self.model_params[x]:
                min_viable_signals = np.add(min_viable_signals, influenced_signal)
            rogue_signals = np.add(rogue_signals, influenced_signal)

        self.data["signal"] = rogue_signals
        self.data["min_viable_signal"] = min_viable_signals
        s = rogue_signals > self.model_params["cut"]
        min_viable_s = min_viable_signals > self.model_params["min_viable_cut"]
        s = np.add(s, min_viable_s) > 0

        num_true_pos, num_false_pos, is_neighbor = self.get_num_true_false(s)
        true_pos = 0
        if self.unstaked:
            num_signal = sum(s)
            print(f"found {num_signal} potential rogues ({100*num_signal/len(s):.2f}")
        else:
            false_pos_mask = s.copy()
            false_pos_mask[self.all_rogue_indices] = False
            false_pos_mask[is_neighbor] = False
            false_pos_mask[
                self.data.loc[
                    ~self.data[AnalysisParameters.ROGUE].str.contains("Normal"), :
                ].index
            ] = False
            if self._hasRogues:
                true_pos = num_true_pos / len(self.rogue_indices)
                false_pos = sum(false_pos_mask) / (
                    len(self.data) - sum(is_neighbor) - len(self.rogue_indices)
                )
                print(
                    "High Confidence Rogues:\n",
                    self.data.loc[
                        self.rogue_indices,
                        ["dataset", "name", "rogue", "signal", "min_viable_signal"],
                    ].to_string(),
                )
                print(40 * "~")
                # indices of other rogues
                print(
                    "Other Rogues:\n",
                    self.data.loc[
                        self.other_rogues,
                        ["dataset", "name", "rogue", "signal", "min_viable_signal"],
                    ].to_string(),
                )
                print(
                    "false positives: \n",
                    self.data.loc[
                        false_pos_mask,
                        ["dataset", "name", "rogue", "signal", "min_viable_signal"],
                    ].to_string(),
                )
                print(40 * "~")
                print(
                    f"{len(self.rogue_indices)} rogues, {len(self.data)-len(self.rogue_indices)} non-rogues"
                )
                print(
                    f"True positives: {100*true_pos:.2f}%, {num_true_pos} / {len(self.rogue_indices)}"
                )
                print(
                    f"False positives: {100*false_pos:.2f}%, {sum(false_pos_mask)} / {len(self.data)-len(self.rogue_indices)}"
                )
                print(
                    f"Other rogues caught: {sum(s[self.all_rogue_indices]) - num_true_pos} / {len(self.all_rogue_indices) - len(self.rogue_indices)}"
                )
                plants_removed = sum(s)
                print(f"Plants removed: {plants_removed} / {len(self.data)}")
                print(
                    f"Purity Before: {1000*(1 - ((len(self.data) - len(self.rogue_indices))/len(self.data))):.2f} [Rogues / 1000]"
                )
                purity = 1 - ((len(self.rogue_indices)) - num_true_pos) / (
                    len(self.data) - plants_removed
                )
                print(f"Purity After:  {1000*(1 - purity):.2f} [Rogues / 1000]")
                self.false_positives = self.data.loc[false_pos_mask, :]
                print(40 * "~")
                if self.log:
                    params = {
                        "rogues": len(self.rogue_indices),
                        "non_rogues": len(self.data) - len(self.rogue_indices),
                        "true_positives": num_true_pos,
                        "true_positives_percent": 100 * true_pos,
                        "false_positives": sum(false_pos_mask),
                        "false_positives_percent": 100 * false_pos,
                        "purity_before": 1000
                        * (
                            1
                            - (
                                (len(self.data) - len(self.rogue_indices))
                                / len(self.data)
                            )
                        ),
                        "purity_after": 1000 * (1 - purity),
                    }
                    mlflow.log_params(params)
            else:
                # only print out locations for false positives
                print(
                    "false positives: \n",
                    self.data.loc[
                        false_pos_mask,
                        ["dataset", "name", "rogue", "signal", "min_viable_signal"],
                    ].to_string(),
                )

        if true_pos != 1:
            print("WARNING: not all rogues were found")
            # find all places in s where there is a rogue but no signal
            self.false_negatives = self.data.loc[self.rogue_indices, :]  # all rogues
            self.false_negatives = self.false_negatives[s[self.rogue_indices] is False]
            print(
                self.false_negatives[
                    ["dataset", "name", "rogue", "signal", "min_viable_signal"]
                ].to_string()
            )

        return true_pos

    def save_model(self):
        """Save the model as a json."""
        for x in self.model_params:  # json can't take np variable types
            if isinstance(self.model_params[x], np.int64):
                self.model_params[x] = int(self.model_params[x])
            if "Stat" in x:
                for z in self.model_params[x]:
                    if isinstance(self.model_params[x][z], np.int64):
                        self.model_params[x][z] = int(self.model_params[x][z])
        with open(f"models/{self.name}_model.json", "w") as f:
            json.dump(self.model_params, f, indent=4)

        if self.log:
            mlflow.log_artifact(f"models/{self.name}_model.json")

    def plot_model_features(self):
        """Plot model features."""
        print("[INFO] Plotting model features")
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        else:
            try:
                os.system(f"rm {self.plot_path}/Stat*")  # remove old plots
            except Exception:
                pass

        plot_tools.plot_2d_mask_cumulative(
            self.data["cumulative-distribution-mask-y-values"],
            self.rogues,
            self.plot_path,
        )
        # plot_tools.plot_centroids(self.data['Centroids-oblique'], self.plot_path, scatter=True, rogues=self.rogues)
        # for feature in self.model_params:
        #     if 'Stats' in feature and self.model_params[feature][ModelParameters.ACTIVE]:
        #         plot_tools.plot_n_lines(
        #             (self.data[feature].values,),
        #             labels = (feature,),
        #             title = f'{feature} for {self.name}',
        #             output_name = f'{self.plot_path}/{feature}',
        #             vlines = self.rogues,
        #             x_labels = self.data[self.features[0]],
        #             _kinternal = not self.public,
        #         )

        #         plot_tools.plot_n_lines(
        #             (1 + (self.data[feature].values/max(self.data[feature].values)),
        #              1 + (self.data[f'{feature}_inf_mean'].values/max(self.data[feature].values)),
        #              self.data[f'{feature}_num_std'].values/max(self.data[f'{feature}_num_std'].values),
        #              [self.model_params[feature][ModelParameters.SIGMA]/max(self.data[f'{feature}_num_std'].values)]*len(self.data[feature].values)
        #                  ),
        #             labels = (f'{feature}', f'mean: $\sigma$ = {self.model_params[feature][ModelParameters.SIGMA]:.2f}, z = {self.model_params[feature][ModelParameters.INFLUENCE]:.4f}', 'num std. devs.','threshold'),
        #             title = f'{feature} with influenced mean, signal, and truth for {self.name}',
        #             output_name = f'{self.plot_path}/{feature}_influenced_mean_signal_truth',
        #             vlines = self.rogues,
        #             x_labels=self.data[self.features[0]],
        #             _kinternal = not self.public,
        #         )

        #         if self.plot_splits:
        #             for split in np.unique(['_'.join(x.split('_')[:2]) for x in self.data[self.features[0]].values]):
        #                 split_plot_path = f'{self.plot_path}/{split}/plots_internal' # no public split plots
        #                 if not os.path.exists(split_plot_path): os.mkdir(split_plot_path)
        #                 split_mask = self.data[self.features[0]].str.contains(split)
        #                 split_data = self.data[split_mask].loc[:, feature].values
        #                 plot_tools.plot_n_lines(
        #                     (split_data,),
        #                     labels = (f'{feature} : {split}',),
        #                     title = f'{feature} for {self.name} {split}',
        #                     output_name = f'{split_plot_path}/{feature}',
        #                     vlines = self.rogues[split_mask],
        #                     x_labels = self.data[self.features[0]][split_mask],
        #                 )
        #             plot_tools()

        # now plot signal, and such
        plot_tools.plot_n_lines(
            (self.data["signal"].values, [self.model_params["cut"]] * len(self.data)),
            labels=("cumulative signal", "threshold"),
            title=f"cumulative signal with trigger threshold for {self.name}",
            output_name=f"{self.plot_path}/signal_with_threshold",
            vlines=self.rogues,
            x_labels=self.data[self.features[0]],
            _kinternal=not self.public,
        )

    def initialize_dynamic_model(self):
        """Initialize the dynamic model."""
        self.dynamic_model_params = {}
        for x in [
            f"{x}_{y}"
            for x in self.features
            for y in AnalysisParameters.STATS_KEYS
            if "Stats" in x and "name" not in y
        ] + [
            "Stats-total-masked-nadir",
            "Stats-cum-dist-oblique_mean",
            "Stats-cum-dist-oblique_d10",
            "Stats-cum-dist-oblique_d30",
            "Stats-cum-dist-oblique_d50",
            "Stats-cum-dist-oblique_d70",
        ]:
            if "kew" in x:
                continue
            if "osis" in x:
                continue
            if "tallest" in x:
                continue
            if "thirty" in x:
                continue
            # if ''
            # if 'nadir' not in x: continue
            # if 'ect' in x: continue
            self.dynamic_model_params[x] = {
                "rtpd": PeakDetector(
                    self.data[: self.window][x].values,
                    self.window,
                    3,  # sigma, TODO come up with a better way to initialize this
                    0.075,  # influence, TODO come up with a better way to initialize this
                ),
                "max_value_seen": max(self.data[: self.window][x].values),
                "num_std_devs": [0] * self.window,
                "signal": [0] * self.window,
                "thresholds": [3] * self.window,
                "influences": [0.25] * self.window,
                "noise_suppressor": [1] * self.window,
                "distance": [0] * self.window,
            }

    def binomcoeffs(self, n):
        """Return the binomial coefficients for a given n."""
        return (np.poly1d([0.5, 0.5]) ** n).coeffs

    def update_peak_detector(
        self,
        rtpd,
        signals,
        std_devs,
        influence_rate=0.015,
        threshold_rate=0.005,
        set_new_max=False,
    ):
        """Update a PeakDetector object."""
        min_influence = 0.001
        max_influence = 0.25
        min_threshold = 2
        max_threshold = 5
        weights = [not x for x in self.model_signals[-10:]]
        avg_std_devs = 1
        # num_signals = 3
        if sum(weights) > 0:
            avg_std_devs = np.average(std_devs[-10:], weights=weights)
            # num_signals = sum(np.multiply(signals[-10:], weights))

        # if the average std deviations is too high, the influence
        #   needs to go up (and vice versa)
        sign_influence = 0 if 0.8 < avg_std_devs < 2 else 1 if avg_std_devs > 2 else -1

        # if the average local noise is too high then the model is
        #   triggering too often and the threshold needs to be raised
        sign_threshold = 0 if 0.5 < avg_std_devs < 5 else -1 if avg_std_devs > 5 else 1

        new_influence = rtpd.influence * abs(
            1 + sign_influence * influence_rate * (1 if sign_influence < 0 else 1)
        )  # allow model to reduce influence faster than increase it
        new_influence = min(max_influence, max(min_influence, new_influence))
        new_threshold = rtpd.threshold * abs(
            1 + sign_threshold * threshold_rate * (1 if sign_threshold > 0 else 1)
        )  # allow model to tighten faster than it loosens
        new_threshold = min(max_threshold, max(min_threshold, new_threshold))
        rtpd.update_params(influence=new_influence, threshold=new_threshold)

    def update_dynamic_model(
        self, new_data: pd.Series, weight: list = None, verbose=False
    ):
        """Update the dynamic model."""
        for x in self.dynamic_model_params:
            # update influence if you see a new max value
            # don't want that to drag around the mean too much
            set_new_max = False
            if new_data[x] > self.dynamic_model_params[x]["max_value_seen"]:
                self.dynamic_model_params[x]["max_value_seen"] = new_data[x]
                set_new_max = True

            self.update_peak_detector(
                self.dynamic_model_params[x]["rtpd"],
                self.dynamic_model_params[x]["signal"][-self.window :],
                self.dynamic_model_params[x]["num_std_devs"][-self.window :],
                set_new_max=set_new_max,
            )

            # update values
            signal, filtered_value, num_std_devs = self.dynamic_model_params[x][
                "rtpd"
            ].thresholding_algo(new_data[x])
            self.dynamic_model_params[x]["thresholds"].append(
                self.dynamic_model_params[x]["rtpd"].threshold
            )
            self.dynamic_model_params[x]["influences"].append(
                self.dynamic_model_params[x]["rtpd"].influence
            )
            self.dynamic_model_params[x]["num_std_devs"].append(num_std_devs)
            self.dynamic_model_params[x]["signal"].append(signal)
            self.dynamic_model_params[x]["distance"].append(
                np.sqrt(np.abs(new_data[x] ** 2 - filtered_value**2))
            )

            # exponentially suppress noisy signals
            num_consecutive_signals = 0
            if signal:
                for i in range(1, self.window):
                    if self.dynamic_model_params[x]["signal"][-i] == 0:
                        break
                    num_consecutive_signals += 1

            self.dynamic_model_params[x]["noise_suppressor"].append(
                self.dynamic_model_params[x]["distance"][-1]
                / self.dynamic_model_params[x]["rtpd"].avgFilter[-1]
            )

    def run_dynamic_model(self):
        """Run the dynamic model.

        The idea is to take the first N plants (start with 30)
        """
        self.stat_error = 1 / np.sqrt(self.window)
        current_position = self.window
        signals = [0] * self.window
        distances = [0] * self.window
        outliers = [0] * self.window
        self.model_signals = [True] * self.window

        thresholds = [0] * self.window
        # influences = [0] * self.window
        # binomial_window = 4
        threshold = 4.5

        # plt.ion()
        # f, ax = plt.subplots(1, 2, figsize=(12,5))
        # line1, = ax[0].plot(smoothed_signals, label='Smoothed Signal (binomial)')
        # line2, = ax[0].plot(filtered_signals, label='Sequential Peak Detector')
        # scatter, = ax[0].plot([], [], marker='.', label='Ground Truth Rogues', c='r', linestyle='None')
        # current_image = f"/home/nschroed/s3_mirror/2023-field-data/Waterman_Strip_Trial/2023-07-18/row-01/104303/DS Splits/DS 000/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/nadir Raw Images/Preprocessed Images/019.png"
        # image = ax[1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(current_image), (256,256)), cv2.COLOR_BGR2RGB))
        # ax[0].legend('upper left')
        # ax[0].set_xlabel('time')
        # ax[1].set_xlabel('Image')
        # ax[0].set_ylabel('signal (smoothed)')

        # run the dynamic model on the data
        while current_position < len(self.data):
            df_current = self.data.iloc[current_position, :]
            self.update_dynamic_model(df_current)

            mean_dist = 0.5
            std_dev_dist = 0.1
            if current_position - self.window * 2 > 0:
                mean_dist = np.average(
                    np.array(distances[-self.window * 2 :])[
                        np.array(outliers[-self.window * 2 :]) < threshold
                    ]
                )
                std_dev_dist = np.std(
                    np.array(distances[-self.window * 2 :])[
                        np.array(outliers[-self.window * 2 :]) < threshold
                    ]
                )

            dist = sum(
                [
                    self.dynamic_model_params[x]["noise_suppressor"][-1] ** 2
                    for x in self.dynamic_model_params
                ]
            ) / len(self.dynamic_model_params)
            distances.append(dist)
            z_score_excluding_outliers = abs(dist - mean_dist) / std_dev_dist
            outliers.append(z_score_excluding_outliers)

            signals.append(
                sum(
                    [
                        self.dynamic_model_params[x]["signal"][-1]
                        * self.dynamic_model_params[x]["noise_suppressor"][-1]
                        for x in self.dynamic_model_params
                    ]
                )
            )
            if signals[-1] > 100:
                signals[-1] = 100
            # trigger if the signal is above the local threshold

            self.model_signals.append(outliers[-1] > threshold)
            thresholds.append(mean_dist + threshold * std_dev_dist)

            # n = 100 if len()
            # line1.set_xdata(range(-len(smoothed_signals)+1,1) if len(smoothed_signals) < 100 else range(-99, 1))
            # line1.set_ydata(smoothed_signals if len(smoothed_signals) < 100 else smoothed_signals[-100:])

            # line2.set_xdata(range(-len(filtered_signals)+1,1) if len(filtered_signals) < 100 else range(-99, 1))
            # line2.set_ydata(filtered_signals if len(filtered_signals) < 100 else filtered_signals[-100:])

            # scatter.set_xdata([x[0]-current_position for x in rogues if current_position >= x[0] > current_position - 100])
            # scatter.set_ydata([x[1] for x in rogues if current_position >= x[0] > current_position - 100])

            # current_image = f"/home/nschroed/s3_mirror/2023-field-data/Waterman_Strip_Trial/2023-07-18/row-01/104303/DS Splits/{df_current['name'].split('_')[0]}/Unfiltered Model Inference/Solo V2 Dec 11 Model (ResNet101-PaddleDetection)/Center Square Crop/nadir Raw Images/Preprocessed Images/{df_current['name'].split('_')[1]}.png"
            # # current_image = current_image.replace(" ", "\ ")
            # img = cv2.resize(cv2.imread(current_image, cv2.IMREAD_COLOR), (512,512))
            # # convert to RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image.set_data(img)

            # [nadir_lines[i].set_xdata(range(len(self.dynamic_model_params[x]['rtpd'].y))) for i,x in enumerate(nadir_params)]
            # [nadir_lines[i].set_ydata(self.dynamic_model_params[x]['rtpd'].y) for i,x in enumerate(nadir_params)]
            # [nadir_filtered_lines[i].set_xdata(range(len(self.dynamic_model_params[x]['rtpd'].filteredY))) for i,x in enumerate(nadir_params)]
            # [nadir_filtered_lines[i].set_ydata(self.dynamic_model_params[x]['rtpd'].filteredY) for i,x in enumerate(nadir_params)]

            # [oblique_lines[i].set_xdata(range(len(self.dynamic_model_params[x]['rtpd'].y))) for i,x in enumerate(oblique_params)]
            # [oblique_lines[i].set_ydata(self.dynamic_model_params[x]['rtpd'].y) for i,x in enumerate(oblique_params)]
            # [oblique_filtered_lines[i].set_xdata(range(len(self.dynamic_model_params[x]['rtpd'].filteredY))) for i,x in enumerate(oblique_params)]
            # [oblique_filtered_lines[i].set_ydata(self.dynamic_model_params[x]['rtpd'].filteredY) for i,x in enumerate(oblique_params)]

            # line2.set_xdata(range(len(smoothed_signals)))
            # line2.set_ydata(smoothed_signals)

            # derivative1.set_xdata(range(len(smoothed_signal_derivative)))
            # derivative1.set_ydata(np.subtract(smoothed_signals, filtered_signals))

            # scatter.set_xdata([x[0] for x in rogues])
            # scatter.set_ydata([x[1] for x in rogues])

            # [axis.relim() for axis in ax]
            # ax[0].relim()
            # ax[0].autoscale_view()
            # ax[0].legend(loc='upper left')
            # [axis.autoscale_view() for axis in ax]
            # [axis.relim() for axis in ax2]
            # [axis.autoscale_view() for axis in ax2]
            # [axis.relim() for axis in ax3]
            # [axis.autoscale_view() for axis in ax3]
            # f.canvas.draw()
            # f.canvas.flush_events()
            # f2.canvas.draw()
            # f2.canvas.flush_events()
            # f3.canvas.draw()
            # f3.canvas.flush_events()

            current_position += 1

        # do some analysis on the data
        self.data["signal"] = signals
        self.data["prediction"] = self.model_signals
        # print(self.get_num_true_false(self.data['signal'].values > 25))
        num_true, num_false, neighbors = self.get_num_true_false(
            self.data["prediction"].values
        )
        # num_true, num_false, neighbors = self.get_num_true_false(np.array(distances) > 0.5)
        other_signals = self.data["prediction"].values[self.other_rogues]

        # print information from df
        print("Rogues:")
        print(
            self.data.loc[
                self.rogue_indices, ["dataset", "name", "rogue", "signal", "prediction"]
            ].to_string()
        )
        print("Other Rogues:")
        print(
            self.data.loc[
                self.other_rogues, ["dataset", "name", "rogue", "signal", "prediction"]
            ].to_string()
        )
        print("False Positives:")
        # drop rogues and other rogues and neighbors from signals
        false_pos = self.data["prediction"].values.copy()
        false_pos[self.rogue_indices] = 0
        false_pos[self.other_rogues] = 0
        false_pos[neighbors] = 0
        false_pos = np.where(false_pos == 1)[0]
        print(
            self.data.loc[false_pos, ["dataset", "name", "rogue", "signal"]].to_string()
        )

        total = len(np.where(np.array(self.model_signals) > 0)[0])
        print(
            f"total signals:    {total} / {len(self.data['prediction'].values)}, {100*total/len(self.data['prediction'].values):.2f}%"
        )
        print(
            f"hybrid rogues:    {num_true} / {len(self.rogue_indices)}, {100*num_true/len(self.rogue_indices):.2f}%"
        )
        print(
            f"hybrid neighbors: {sum(neighbors)} / {sum(self.data['prediction'].values)}, {100*sum(neighbors)/len(self.data['prediction'].values):.2f}%"
        )
        if len(self.other_rogues) > 0:
            print(
                f"other rogues:     {sum(other_signals)} / {len(other_signals)}, {100*sum(other_signals)/len(other_signals):.2f}%"
            )
        print(
            f"false positives:  {num_false} / {len(self.data) - len(self.rogue_indices)}, {100*num_false/(len(self.data) - len(self.rogue_indices)):.2f}%"
        )

        # plot signal
        plot_tools.plot_n_lines(
            (signals, -1 * self.data["prediction"].values),
            labels=("cumulative weighted signal", "prediction"),
            title=f"cumulative signal with trigger threshold for {self.name}",
            output_name=f"{self.plot_path}/signal_with_threshold",
            vlines=self.rogues,
            x_labels=self.data[self.features[0]],
            # log_y=True,
        )

        plot_tools.plot_n_lines(
            (distances, thresholds),
            labels=("cumulative relative distance", "threshold"),
            title=f"cumulative relative distance with trigger threshold for {self.name}",
            output_name=f"{self.plot_path}/distance_with_threshold",
            vlines=self.rogues,
            x_labels=self.data[self.features[0]],
            log_y=True,
        )

        # plot distances using plot tools plot_n_lines
        plot_tools.plot_n_lines(
            (
                outliers,
                [threshold for _ in outliers],
            ),
            labels=("distance z-score", f"threshold = {threshold}"),
            title=f"Distance Z-Score for {self.name}",
            output_name=f"{self.plot_path}/signal_distance_z_score",
            vlines=self.rogues,
            x_labels=self.data[self.features[0]],
            log_y=True,
        )

        # plot rogue locations as dotted lines
        for x in self.dynamic_model_params:
            if x != "Stats-rectangular-oblique_max":
                continue
            upper_threshold = np.add(
                self.dynamic_model_params[x]["rtpd"].avgFilter,
                np.multiply(
                    self.dynamic_model_params[x]["rtpd"].stdFilter,
                    self.dynamic_model_params[x]["rtpd"].threshold,
                ),
            )
            plot_tools.plot_n_lines(
                (
                    self.data[x].values,
                    self.dynamic_model_params[x]["rtpd"].avgFilter,
                    upper_threshold,
                ),
                labels=(f"{x}", "Influenced Mean", "Threshold for using influence"),
                title=f"{x} with influenced mean and upper threshold for {self.name}",
                output_name=f"{self.plot_path}/{x}",
                vlines=self.rogues,
                x_labels=self.data[self.features[0]],
            )
            # plot noise_suppressor for each metric
            plot_tools.plot_n_lines(
                (self.dynamic_model_params[x]["influences"],),
                labels=("influence",),
                title=f"influence for {x} for {self.name}",
                output_name=f"{self.plot_path}/{x}_influence",
                vlines=self.rogues,
                x_labels=self.data[self.features[0]],
            )
            plot_tools.plot_n_lines(
                (self.dynamic_model_params[x]["thresholds"],),
                labels=("threshold",),
                title=f"threshold for {x} for {self.name}",
                output_name=f"{self.plot_path}/{x}_threshold",
                vlines=self.rogues,
                x_labels=self.data[self.features[0]],
            )
            plot_tools.plot_n_lines(
                (self.dynamic_model_params[x]["distance"],),
                labels=("distance",),
                title=f"distance from influenced mean for {x} for {self.name}",
                output_name=f"{self.plot_path}/{x}_distance",
                vlines=self.rogues,
                x_labels=self.data[self.features[0]],
            )
            plot_tools.plot_n_lines(
                (self.dynamic_model_params[x]["noise_suppressor"],),
                labels=("weight (noise suppressor)",),
                title=f"noise suppressor for {x} for {self.name}",
                output_name=f"{self.plot_path}/{x}_noise_suppressor",
                vlines=self.rogues,
                x_labels=self.data[self.features[0]],
            )
