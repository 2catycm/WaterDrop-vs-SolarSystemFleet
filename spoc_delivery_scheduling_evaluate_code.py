# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import numpy as np

# --------------------------------------
import os

# --------------------------------------
import json

# --------------------------------------
import matplotlib.pyplot as plt


class trappist_schedule:
    """
    UDP (User-Defined Problem) for the Trappist-1 scheduling optimisation problem.
    This corresponds to the third challenge in SpOC (Space Optimisation Competition)
    conceived by the ACT for the GECCO 2022 conference.

    The challenge involves optimising the schedule for delivering asteroids to
    hypothetical processing stations in a differnt orbit in the far future.
    A more detailed overview of the problem scenario and the three challenges can be found here:

    https://www.esa.int/gsp/ACT/projects/gecco-2022-competition/

    This class conforms to the pygmo UDP format.
    """

    def __init__(
        self,
        path=os.path.join(".", "data", "spoc", "scheduling", "candidates.txt"),
        n_stations=12,
        start_time=0.0,
        end_time=80.0,
        station_gap=1.0,
        materials=(
            "Material A",
            "Material B",
            "Material C",
        ),
    ):
        # Database of asteroid-to-station visit opportunities,
        self.db = self._load(path)

        # Number of stations
        self.n_stations = n_stations

        # Number of asteroids
        self.n_asteroids = len(self.db)

        # A flattened version of the database with
        # (asteroid ID, station ID, opportunity ID) tuples as keys and
        # (arrival time, mass A, mass B, mass C) tuples as values.
        # This method also computes the maximum number of opportunities
        # in the database (relevant for bounds checks)
        (self.flat_db, self.max_opportunities) = self._flatten(self.db)

        # The start and end times for the whole problem.
        # Units: days
        self.start_time = start_time
        self.end_time = end_time

        # Station gap (minimum time allowed between activating two consecutive stations).
        # Units: days
        self.station_gap = station_gap

        # List of material names
        self.materials = materials

    def get_nobj(self):
        """
        There is only one objective for this challenge:
        to maximise the minimum amount of material collected per station.

        Returns:
            Number of objectives.
        """
        return 1

    def get_nix(self):
        """
        Each assignment consists of a pair of asteroid ID and station ID,
        hence the total number is 2 x the number of asteroids.

        Returns:
            Number of integer components of the chromosome.
        """
        return self.n_asteroids * 2

    def get_nec(self):
        """
        There are two equality constraints (cf. _fitness_impl() for details).

        Returns:
            Number of equality constraints.
        """
        return 2

    def get_nic(self):
        """
        There are two equality constraints
        (cf. _fitness_impl() for details).

        Returns:
            Number of inequality constraints.
        """
        return 2

    def get_bounds(self):
        """
        Bounds for chromosome elements.

        Returns:
            Bounds for each element in the chromosome.
        """

        lb = [self.start_time] * (2 * self.n_stations)
        lb.extend([1, 0, 0] * self.n_asteroids)

        ub = [self.end_time] * (2 * self.n_stations)
        ub.extend(
            [self.n_asteroids, self.n_stations, self.max_opportunities]
            * self.n_asteroids
        )

        return (lb, ub)

    def _load(
        self,
        path,
    ):
        """

        Load the database from an external JSON file.

        Args:
            path: The path to the database file

        Returns:
            The path to the database file.
        """

        with open(path) as db:
            _db = json.loads(db.read())

        db = {}
        for ast_id, stations in _db.items():

            opportunities = {}
            for stat_id, opps in stations.items():
                # Convert the station ID from str to int.
                opportunities[int(stat_id)] = list(opps)

            # Convert the asteroid ID from str to int.
            db[int(ast_id)] = opportunities

        return db

    def _flatten(
        self,
        db,
    ):

        """
        Flatten the database.

        Args:
            db: The database of possible asteroid / station assignment opportunities.

        Returns:
            A flat version of the database with (asteroid ID, station ID, opportunity ID)
            tuples as keys and (arrival time, mass A, mass B, mass C) tuples as values and
            the maximum number of opportunities for any asteroid / station pair in the database.
        """

        flat_db = {}
        max_opps = 0
        for ast_id, stations in db.items():
            for stat_id, opps in stations.items():
                if len(opps) > max_opps:
                    max_opps = len(opps)
                for idx, opp in enumerate(opps):
                    flat_db[(ast_id, stat_id, idx)] = opp

        return (flat_db, max_opps - 1)

    def _plot(
        self,
        masses,
        schedule,
        ax=None,
        path=None,
    ):
        """
        Plot the total material masses at each station and
        the schedule vs. opportunities for each station.

        Args:
            masses: A 2D array containing the masses corresponding to all assignment opportunities.
            schedule: The actual scheduled asteroid / station assignments and their corresponding masses.
            ax: Plot axes. Defaults to None.
            path: A file to save the plot to. Defaults to None.

        Returns:
            Plot axes.
        """

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 18))

        (m_ax, w_ax) = ax[0], ax[1]

        # ==[ Plot mass distribution ]==

        indices = np.arange(1, self.n_stations + 1)
        bar_width = 0.2

        m_ax.bar(
            indices - bar_width,
            masses[:, 0],
            bar_width,
            color="r",
            label="Material A",
        )
        m_ax.bar(
            indices,
            masses[:, 1],
            bar_width,
            color="g",
            label="Material B",
        )
        m_ax.bar(
            indices + bar_width,
            masses[:, 2],
            bar_width,
            color="b",
            label="Material C",
        )

        # ==[ Plot minimum masses for each material]==
        min_masses = masses.min(axis=0)
        m_ax.plot(
            [0, self.n_stations + 1],
            [min_masses[0], min_masses[0]],
            "r--",
            label=f"Minimum mass of {self.materials[0]}",
        )
        m_ax.plot(
            [0, self.n_stations + 1],
            [min_masses[1], min_masses[1]],
            "g--",
            label=f"Minimum mass of {self.materials[1]}",
        )
        m_ax.plot(
            [0, self.n_stations + 1],
            [min_masses[2], min_masses[2]],
            "b--",
            label=f"Minimum mass of {self.materials[2]}",
        )

        m_ax.set_xlim((0.4, 12.6))
        m_ax.set_ylim((0.4, masses.max() + 4))
        m_ax.set_xticks(list(range(1, self.n_stations + 1)))
        m_ax.set_xticklabels(list(range(1, self.n_stations + 1)))
        m_ax.set_xlabel("Station")
        m_ax.set_ylabel("Material masses")
        m_ax.legend()

        # ==[ Plot schedule ]==

        # Plot all opportunities
        for stat_id, data in schedule.items():

            if len(data) == 0:
                continue

            opportunities, atimes, window = (
                data["opportunities"],
                data["assignments"],
                data["window"],
            )

            # Opportunities
            w_ax.plot(
                opportunities,
                np.ones((len(opportunities),)) * stat_id,
                "r.",
                ms=1,
                label="Opportunities" if stat_id == self.n_stations else None,
            )

            # Arrival times
            w_ax.plot(
                atimes,
                np.ones((len(atimes),)) * stat_id,
                "c|",
                ms=6,
                label="Assignments" if stat_id == self.n_stations else None,
            )

            # Window
            w_ax.plot(
                [window[0], window[0]],
                [1, self.n_stations],
                "--",
                color="lightgray",
                lw=0.5,
            )
            w_ax.plot(
                [window[1], window[1]],
                [1, self.n_stations],
                "--",
                color="darkgray",
                lw=0.5,
            )

        w_ax.plot([self.start_time, self.start_time], [1, self.n_stations], "k-")
        w_ax.plot([self.end_time, self.end_time], [1, self.n_stations], "k-")

        w_ax.set_xlabel("Time [days]")

        w_ax.set_yticks(list(range(1, self.n_stations + 1)))
        w_ax.set_yticklabels(list(range(1, self.n_stations + 1)))
        w_ax.set_ylabel("Station")
        w_ax.set_ylim((0, 14))

        w_ax.legend(loc=1)

        if path is not None:
            fig.savefig(path, dpi=100)

        return ax

    def _fitness_impl(
        self,
        x,
        logging=False,
        plotting=False,
        ax=None,
        path=None,
    ):
        """
        Computes the constraints and the fitness of the provided chromosome.

        1. Equality constraints:

        1.1. Asteroid IDs: all asteroids in the database must be present in the chromosome.
        1.2. Opportunity IDs: all opportunity IDs in the chromosome must correspond to opportunities in the database.

        2. Inequality constraints:

        2.1. Station gaps: all station gaps must be greater than a minimal time period (self.station_gap)
        2.2. Arrival times: all asteroid arrival times must be between the start and end times of the corresponding station

        3. Fitness:

        3.1 Iterate over the chromosome and add the masses of the materials for all assigned asteroids with valid transfers
        3.2 Find the minimum mass of each material per station. This is the final fitness.

        Args:
            x: A list of integers and floats in the following format:
                - Station start and end times (2 x self.n_stations)
                - The following items for all selected asteroids (3 x self.n_asteroids integers in total):
                    - Asteroid ID
                    - Station ID
                    - Opportunity ID

                NOTE: The triplets do not have to be ordered by asteroid ID.

            logging: Logging switch. Defaults to False.
            plotting: Plotting switch. Defaults to False.
            ax: Plot axes. Defaults to None.
            path: File path for saving the plots. Defaults to None.

        Returns:
            A tuple containing:
                - The fitness
                - A list of equality constraints
                - A list of inequality constraints
                - Plot axes

        """

        all_violations = 0

        eq_constraints = []
        ineq_constraints = []

        # Offset representing the time windows for stations (for convenience only)
        station_times_offset = 2 * self.n_stations

        # Extract a set of tuples of all the selections as tuples
        asteroid_ids = [int(a_id) for a_id in x[station_times_offset::3]]
        station_ids = [int(s_id) for s_id in x[(station_times_offset + 1) :: 3]]
        assignments = [
            int(assignment - 1) for assignment in x[(station_times_offset + 2) :: 3]
        ]

        triplets = tuple(zip(asteroid_ids, station_ids, assignments))

        # Extract the start and end times for all stations from the chromosome
        station_times = x[:station_times_offset]
        station_start_times = station_times[0 : len(station_times) : 2]
        station_end_times = station_times[1 : len(station_times) : 2]

        """
        1. Equality constraints.
        """

        # ==[ 1.1 Check asteroid IDs ]==

        asteroid_id_violations = set(asteroid_ids).symmetric_difference(
            set(self.db.keys())
        )

        eq_constraints.append(len(asteroid_id_violations))
        all_violations += len(asteroid_id_violations)

        # ==[ 1.2. Check opportunities ]==

        # Check the opportunity IDs  for asteroid / station assignments
        # where the station ID is *not* 0 (i.e., the asteroid has
        # been activated and selected for transfer to a valid station).
        opportunity_id_violations = [
            triplet
            for triplet in triplets
            if triplet not in self.flat_db and triplet[1] > 0
        ]

        eq_constraints.append(len(opportunity_id_violations))
        all_violations += len(opportunity_id_violations)

        """
        2. Inequality constraints.
        """

        # First we sort and index station activation windows.
        # We produce a list where each entry has the following format:
        #
        # [idx, [start, end]]
        #
        # where
        # - idx: index in the *original* chromosome (= station ID)
        # - (start, end): start and end times for visiting the corresponding station

        station_windows = zip(station_start_times, station_end_times)
        sorted_indexed_windows = sorted(
            enumerate(station_windows, 1), key=lambda i: i[1]
        )

        # ==[ 2.1. Check station gaps ]==

        # Compute the gaps between stations from the sorted array of station windows
        gaps = [
            s2[1][0] - s1[1][1]
            for s1, s2 in zip(sorted_indexed_windows[:-1], sorted_indexed_windows[1:])
        ]

        gap_violations = np.array(self.station_gap - np.array(gaps, dtype=np.float32))
        ineq_constraints.append(gap_violations.max())
        all_violations += np.count_nonzero(np.where(gap_violations > 0.0, 1, 0))

        # ==[ 2.2. Arrival times ]==

        # Collect all asteroid times for each station
        station_window_start_times = []
        station_window_end_times = []

        for s_id in station_ids:
            if s_id == 0:
                # Trick to make sure that unassigned asteroids don't trigger
                # an inequality constraint violation
                station_window_start_times.append(-1.0)
                station_window_end_times.append(self.end_time + 1.0)
            else:
                station_window_start_times.append(station_start_times[s_id - 1])
                station_window_end_times.append(station_end_times[s_id - 1])

        # Find any instances where the arrival time is before the start time or
        # after the end time of the corresponding station's window.
        arrival_times = np.array(
            [
                self.flat_db[triplet][0]
                if triplet in self.flat_db
                else (self.end_time - self.start_time) / 2
                for triplet in triplets
            ],
            dtype=np.float32,
        )
        station_window_start_times = np.array(
            station_window_start_times, dtype=np.float32
        )
        station_window_end_times = np.array(station_window_end_times, dtype=np.float32)

        # Find instances where the arrival times are earier than the
        # start time for the corresponding station
        arrival_time_violations = np.where(
            arrival_times - station_window_start_times < 0.0, 1, 0
        )

        # Add to those any instances where the arrival times are later than the
        # end time for the corresponding station
        arrival_time_violations += np.where(
            station_window_end_times - arrival_times < 0.0, 1, 0
        )

        # Find any violations (1s indicate a violation in either start time or end time)
        arrival_time_violations = np.where(arrival_time_violations > 0, 1, 0)
        ineq_constraints.append(arrival_time_violations.sum())
        all_violations += arrival_time_violations.sum()

        """
        3. Fitness
        """

        fitness = 0.0
        if all_violations == 0:

            # Compute the masses of all materials accumulated at each station
            masses_per_station = {
                s_id: np.array([0.0, 0.0, 0.0], dtype=np.float32)
                for s_id in range(1, self.n_stations + 1)
            }

            for (triplet, violation) in zip(
                triplets, arrival_time_violations.astype(bool).tolist()
            ):
                # If the asteroid is assigned to a valid station and
                # its arrival time is within bounds...
                if triplet[1] > 0 and not violation:
                    # Add the material masses to the stations
                    masses_per_station[triplet[1]] += np.array(
                        self.flat_db[triplet][1:],
                        dtype=np.float32,
                    )

            # Collect all the masses per station into a single 2D array that is easy to manipulate
            masses = np.array(
                [masses_per_station[s] for s in range(1, self.n_stations + 1)]
            )

            # Final fitness computation.
            # The objective is to maximise the minimum mass
            # of any material across all stations.
            fitness = -masses.min()

        if logging:

            print(
                f"==[ Invalid asteroid IDs: {eq_constraints[0]} out of {len(self.db)}"
            )
            print(f"==[ Invalid arrival times: {eq_constraints[1]}")
            print(f"==[ Minimal inter-station gap: {min(gaps):<2.4}")
            print(f"==[ Invalid assignments: {ineq_constraints[1]}")
            print(f"==[ Masses per station:")
            print(
                f"{'Station ID':>12} {self.materials[0]:>12} {self.materials[1]:>12} {self.materials[2]:>12}"
            )
            for stat_id, mass_dist in masses_per_station.items():
                print(
                    f"{stat_id:>12} {mass_dist[0]:>12.6f} {mass_dist[1]:>12.6f} {mass_dist[2]:>12.6f}"
                )

            for idx, item in enumerate(gap_violations):
                if item > 0:
                    # Get the station IDs in the original chromosome
                    station_1_id = sorted_indexed_windows[idx][0]
                    station_2_id = sorted_indexed_windows[idx + 1][0]
                    gap = gaps[idx]
                    if gap <= self.station_gap:
                        print(
                            f"==[\tThe gap between stations {station_1_id} and {station_2_id} is {gap:3.3f} (should be >= {self.station_gap:3.3f})."
                        )
                    else:
                        print(
                            f"==[\tThe windows for stations {station_1_id} and {station_2_id} overlap by {-gap:3.3f} days."
                        )

            print(f"==[ Total fitness: {fitness}")

        # Plotting
        if plotting:
            schedule = {
                s_id: {
                    "opportunities": [],
                    "assignments": [],
                    "window": [],
                }
                for s_id in range(1, self.n_stations + 1)
            }

            for (_, s_id, _), val in self.flat_db.items():
                schedule[s_id]["opportunities"].append(val[0])

            for triplet in triplets:
                if triplet[1] > 0:
                    schedule[triplet[1]]["assignments"].append(self.flat_db[triplet][0])

            for s_id, window in sorted_indexed_windows:
                if s_id > 0:
                    schedule[s_id]["window"] = window

            ax = self._plot(masses, schedule, ax=ax, path=path)

        return (fitness, eq_constraints, ineq_constraints, ax)

    def fitness(
        self,
        x,
    ):
        """
        A wrapper for the fitness function (called for evaluation only).

        #################### IMPORTANT ######################
        - The chromosome has the following format:

            - Start and end times for each station, *in order of Station ID*
            - Asteroid / station assignments with the corresponding arrival times.

            Format:
            [
                Station 1 start time, Station 1 end time,   |
                Station 2 start time, Station 2 end time,   |
                ...                                         | 2 x number of stations
                Station 11 start time, Station 11 end time, |
                Station 12 start time, Station 12 end time, |
                Asteroid ID, Station ID, Opportunity ID, |
                Asteroid ID, Station ID, Opportunity ID, |
                ...                                      | number of asteroids
                Asteroid ID, Station ID, Opportunity ID, |
                Asteroid ID, Station ID, Opportunity ID  |
            ]

        - All IDs (for asteroids, stations and opportunities) are 1-based.
            - This is particularly relevant for the opportunity IDs
            since they are converted to 0-based indices in the fitness
            evaluation function by subtracting 1.

        - Stations must be activated *sequentially* (*not* in parallel) but not necessarily in order of their ID.

        - There must be a minimal gap (called 'station gap') between the end time of one station
        and the start time of the next.

        - Every asteroid must be either asigned to a station or unassigned.

        - The asteroid / station assignments do not have to be  in any particular order,
        but all asteroid IDs must appear in the chromosome, even if some asteroids are unassigned.

            - Assigned asteroids must have corresponding Station IDs between 1 and the number of stations.
            - Unassigned asteroids must have a Station ID 0.

        ######################################################

        Args:
            x: A chromosome in the format specified above.

        Returns:
            A tuple containing the fitness followed by the equality and inequality constraints.

        """

        (fitness, eq_constraints, ineq_constraints, ax) = self._fitness_impl(x)

        return (fitness, *eq_constraints, *ineq_constraints)

    def pretty(
        self,
        x,
    ):
        """
        Fitness evaluation function with pretty printing.

        Args:
            x: A chromosome.
        """

        (_, _, _, ax) = self._fitness_impl(x, logging=True)

    def plot(
        self,
        x,
        ax=None,
        path=None,
    ):
        """
        Plot the total material masses accumulated at each station
        and the asteroid / station assignments.

        Args:
            x: A chromosome.
            ax: Plot axes. Defaults to None.
            path: A file to save the plot to. Defaults to None.

        Returns:
            Plot axes.
        """
        (_, _, _, ax) = self._fitness_impl(
            x,
            logging=False,
            plotting=True,
            ax=ax,
            path=path,
        )

        return ax

    def example(self):
        """
        An example method returning a minimal chromosome that assigns
        a single asteroid to each station.

        Returns:
            A valid chromosome.
        """

        assignments = {}

        # Iterate over a random permutation of asteroid IDs
        for ast_id in list(self.db.keys()):

            if len(assignments) == self.n_stations:
                break

            # Create a set of station IDs.
            stations = set(self.db[ast_id].keys()).difference(set(assignments.keys()))
            while len(stations) > 0:
                stat_id = stations.pop()

                # Check if the asteroid / station pair
                # is already assigned.
                if stat_id in assignments:
                    break

                opps = self.db[ast_id][stat_id]
                # Check if there are any opportunities
                for opp_idx, opp in enumerate(opps, 1):
                    # Check if the arrival time of the opportunity conflicts with
                    # the assigned asteroid / station build times
                    conflict = False
                    for _, vals in assignments.items():
                        if ast_id == vals[0] or (vals[2] - 2 * self.station_gap) <= opp[
                            0
                        ] <= (vals[3] + 2 * self.station_gap):
                            conflict = True
                            break
                    if not conflict:
                        assignments[stat_id] = [
                            ast_id,
                            opp_idx,
                            opp[0] - self.station_gap,
                            opp[0] + self.station_gap,
                        ]
                        stations = set()
                        break

        # Build the chromosome
        windows = []
        schedule = []
        for stat_id in range(1, self.n_stations + 1):
            # Station start time
            windows.append(assignments[stat_id][2])
            # Station end time
            windows.append(assignments[stat_id][3])
            # Asteroid / station assignment
            schedule.extend([assignments[stat_id][0], stat_id, assignments[stat_id][1]])

        # Stitch the windows and schedule together and create
        # a complete chromosome from this partial solution.
        chromosome = self.convert_to_chromosome(
            np.concatenate((np.array(windows), np.array(schedule)))
        )

        return chromosome

    def convert_to_chromosome(
        self,
        x,
    ):
        """
        Creates a valid chromosome from an incomplete one.

        Here, 'incomplete' means that all station windows are provided
        but only some asteroids are assigned. This method completes the
        chromosome by assigning the missing asteroids to station 0, which
        means that those asteroids will not be considered in the fitness evaluation.

        Args:
            x: Incomplete chromosome.

        Returns:
            Complete chromosome.
        """

        if len(x) < 2 * self.n_stations:
            raise ValueError(
                "The chromosome must contain at least the start and end times for the station windows."
            )

        assignments = list(x[24:])

        # Check if we have any asteroids assigned at all
        if len(assignments) > 0:
            assignments = {
                assignment[0]: assignment
                for assignment in zip(
                    assignments[::3], assignments[1::3], assignments[2::3]
                )
            }

        schedule = []
        for ast_id in range(1, self.n_asteroids + 1):
            if ast_id not in assignments:
                schedule.extend([ast_id, 0, 0])
            else:
                schedule.extend(assignments[ast_id])

        return np.concatenate((np.array(x[:24]), np.array(schedule)))


udp = trappist_schedule()
