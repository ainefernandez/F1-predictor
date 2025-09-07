import fastf1
import pandas as pd
from fastf1.ergast import Ergast

fastf1.Cache.enable_cache("f1_cache")


def get_driver_standings(season, round_limit):
    """
    Calculate driver standings up to a specific round in a given season.

    Returns:
        standings (pd.DataFrame): DataFrame with columns ['Driver', 'Points', 'ChampPosition']
    """
    ergast = Ergast()
    results = []
    races = ergast.get_race_schedule(season=season)

    for rnd, race in races['raceName'].items():
        if rnd + 1 > round_limit:
            break

        temp = ergast.get_race_results(season=season, round=rnd + 1)
        temp = temp.content[0]

        sprint = ergast.get_sprint_results(season=season, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            temp = pd.merge(temp, sprint.content[0], on='driverCode', how='left', suffixes=('_race', '_sprint'))
            temp['points'] = temp['points_race'] + temp['points_sprint'].fillna(0)
            temp.drop(columns=['points_race', 'points_sprint'], inplace=True)
        else:
            temp['points'] = temp['points']

        temp['round'] = rnd + 1
        temp['race'] = race.removesuffix(' Grand Prix')
        temp = temp[['round', 'race', 'driverCode', 'points']]
        results.append(temp)

    results = pd.concat(results, ignore_index=True)
    results_wide = results.pivot(index='driverCode', columns='round', values='points')
    results_wide['total_points'] = results_wide.sum(axis=1)
    results_wide = results_wide.sort_values(by='total_points', ascending=False)

    standings = results_wide['total_points'].reset_index()
    standings.columns = ['Driver', 'Points']
    standings['ChampPosition'] = standings['Points'].rank(method='min', ascending=False).astype(int)
    standings = standings.sort_values(by='ChampPosition')

    return standings


def get_fp_data(season, race, fpsession):
    """
    Retrieves Free Practice data (FP1/FP2/FP3) for a given season and race.

    Returns a DataFrame with:
        Driver, Team, GapToFastest, Position, RacePace
    """
    fp = fastf1.get_session(season, race, fpsession)
    fp.load()

    laps = fp.laps.copy()
    laps["LapTime"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTime"])

    fastest_laps = laps.groupby("Driver").apply(lambda x: x.nsmallest(1, "LapTime")).reset_index(drop=True)
    fp_data = pd.DataFrame()
    fp_data["Driver"] = fastest_laps["Driver"].values

    results = fp.results[["Abbreviation"]].copy()
    results.rename(columns={"Abbreviation": "Driver"}, inplace=True)
    fp_data = fp_data.merge(results, on="Driver", how="left")

    fp_fastest_time = fastest_laps["LapTime"].min()
    fp_data[f"GapToFastest_{fpsession}"] = fastest_laps["LapTime"].values - fp_fastest_time
    fp_data[f"Position_{fpsession}"] = fp_data[f"GapToFastest_{fpsession}"].rank(method="min").astype(int)

    quicklaps = laps.pick_quicklaps()
    racepacemean = quicklaps.groupby("Driver")["LapTime"].mean().reset_index()
    fp_data = fp_data.merge(racepacemean, on="Driver", how="left")
    fp_data.rename(columns={"LapTime": f"RacePace_{fpsession}"}, inplace=True)
    fp_data = fp_data.sort_values(by=f"GapToFastest_{fpsession}", ascending=True).reset_index(drop=True)

    return fp_data


def get_quali_data(season, race):
    """
    Get qualifying data with the latest lap per driver and gap to pole.

    Uses Q3 time for Q3 drivers, Q2 for Q2 dropouts, Q1 for Q1 dropouts.
    Returns Driver, Position_Quali, QualiLap, GapToPole
    """
    quali = fastf1.get_session(season, race, "Q")
    quali.load()
    results = quali.results.copy()

    for i in ["Q1", "Q2", "Q3"]:
        results[i] = results[i].dt.total_seconds()

    results["QualiLap"] = results["Q3"].fillna(results["Q2"]).fillna(results["Q1"])
    q3_times = results["Q3"].dropna()
    pole_time = q3_times.min()
    results["GapToPole"] = results["QualiLap"] - pole_time

    quali_data = results[["Abbreviation", "Position", "QualiLap", "GapToPole"]].copy()
    quali_data.rename(columns={"Abbreviation": "Driver", "Position": "Position_Quali"}, inplace=True)
    quali_data = quali_data.sort_values(by="Position_Quali").reset_index(drop=True)

    return quali_data


def get_race_data(season, race):
    """
    Retrieves race data for a given season and race.

    Returns a DataFrame with:
        Driver, FinishingPosition, FinishStatus, DNF_Dummy,
        RacePace, GapToFastestRacePace, RaceTime
    """
    session = fastf1.get_session(season, race, "R")
    session.load()
    laps = session.laps.copy()
    laps["LapTime"] = laps["LapTime"].dt.total_seconds()

    quicklaps = laps.pick_quicklaps()
    race_pace = quicklaps.groupby("Driver")["LapTime"].mean().reset_index()
    race_pace.rename(columns={"LapTime": "RacePace"}, inplace=True)
    fastest_race_pace = race_pace["RacePace"].min()
    race_pace["GapToFastestRacePace"] = race_pace["RacePace"] - fastest_race_pace

    results = session.results[["Abbreviation", "Position", "Status", "Time"]].copy()
    results.rename(columns={"Abbreviation": "Driver", "Position": "FinishingPosition"}, inplace=True)
    results["Time"] = results["Time"].dt.total_seconds()
    results["DNF_Dummy"] = results["Time"].isna().astype(int)

    def simplify_status(s):
        s = str(s)
        if "Finished" in s:
            return "Finished"
        elif "Lap" in s:
            return "Lapped"
        else:
            return "DNF"

    results["FinishStatus"] = results["Status"].apply(simplify_status)
    winner_time = results.loc[results["FinishingPosition"] == 1, "Time"].values[0]

    def compute_racetime(row):
        if row["DNF_Dummy"] == 1:
            return None

        driver = row["Driver"]
        driver_laps_all = laps[laps["Driver"] == driver]
        completed_laps = driver_laps_all.shape[0]
        total_laps = laps["LapNumber"].max()
        missing_laps = total_laps - completed_laps
        driver_quicklaps = quicklaps[quicklaps["Driver"] == driver]
        avg_quicklap = driver_quicklaps["LapTime"].mean()
        base_time = winner_time + row["Time"] if row["FinishingPosition"] != 1 else winner_time

        if "Lap" in str(row["Status"]):
            return base_time + avg_quicklap * missing_laps
        return base_time

    results["RaceTime"] = results.apply(compute_racetime, axis=1)
    race_data = results.merge(race_pace, on="Driver", how="left")
    race_data = race_data.drop("Time", axis=1)
    race_data = race_data.sort_values(by="FinishingPosition").reset_index(drop=True)

    return race_data


def get_season_data_with_features(season, round_limit):
    """
    Build season-level dataset including FP, Quali, Race data
    and rolling season-to-date features for each driver.

    Returns a DataFrame with all rounds concatenated.
    """
    all_rounds = []
    cumulative_data = []

    mapTeams = {
        "NOR": "McLaren", "PIA": "McLaren",
        "VER": "Red Bull", "TSU": "Red Bull",
        "HAM": "Ferrari", "LEC": "Ferrari",
        "RUS": "Mercedes", "ANT": "Mercedes",
        "ALB": "Williams", "SAI": "Williams",
        "STR": "Aston Martin", "ALO": "Aston Martin",
        "HAD": "Racing Bulls", "LAW": "Racing Bulls",
        "HUL": "Kick Sauber", "BOR": "Kick Sauber",
        "BEA": "Haas", "OCO": "Haas",
        "GAS": "Alpine", "COL": "Alpine"
    }

    for rnd in range(1, round_limit + 1):
        fp_data = []
        for fp in ["FP1", "FP2", "FP3"]:
            try:
                fp_df = get_fp_data(season, rnd, fp)
                fp_data.append(fp_df)
            except Exception as e:
                print(f"⚠️ Could not load {fp} for round {rnd}: {e}")

        fp_merged = fp_data[0] if fp_data else pd.DataFrame(columns=["Driver"])
        for df in fp_data[1:]:
            fp_merged = fp_merged.merge(df, on="Driver", how="outer")

        try:
            quali_data = get_quali_data(season, rnd)
        except Exception as e:
            print(f"⚠️ Could not load Quali for round {rnd}: {e}")
            quali_data = pd.DataFrame(columns=["Driver"])

        try:
            race_data = get_race_data(season, rnd)
        except Exception as e:
            print(f"⚠️ Could not load Race for round {rnd}: {e}")
            race_data = pd.DataFrame(columns=["Driver"])

        round_data = race_data
        if not quali_data.empty:
            round_data = round_data.merge(quali_data, on="Driver", how="outer")
        if not fp_merged.empty:
            round_data = round_data.merge(fp_merged, on="Driver", how="outer")
        round_data["Round"] = rnd

        cumulative_data.append(round_data)
        season_to_date = pd.concat(cumulative_data, ignore_index=True)

        if season_to_date.empty or "Driver" not in season_to_date.columns:
            rolling_features = pd.DataFrame(columns=["Driver"])
        else:
            rolling_features = season_to_date.groupby("Driver").agg(
                AvgGapToPole=("GapToPole", "mean"),
                AvgFinishingPosition=("FinishingPosition", "mean"),
                AvgQualiPosition=("Position_Quali", "mean"),
                AvgFP1Position=("Position_FP1", "mean"),
                AvgFP2Position=("Position_FP2", "mean"),
                AvgFP3Position=("Position_FP3", "mean"),
                AvgGaptofastestRacePace=("GapToFastestRacePace", "mean"),
                DNFPercentage=("DNF_Dummy", "mean"),
                AvgFP1Gap=("GapToFastest_FP1", "mean"),
                AvgFP2Gap=("GapToFastest_FP2", "mean"),
                AvgFP3Gap=("GapToFastest_FP3", "mean")
            ).reset_index()

            rolling_features["Team"] = rolling_features["Driver"].map(mapTeams)

            try:
                standings = get_driver_standings(season, rnd)
                rolling_features = rolling_features.merge(standings, on="Driver", how="left")
            except Exception as e:
                print(f"⚠️ Could not load standings for round {rnd}: {e}")

        round_with_features = round_data.merge(rolling_features, on="Driver", how="left") if not rolling_features.empty else round_data.copy()
        all_rounds.append(round_with_features)

    full_season_data = pd.concat(all_rounds, ignore_index=True)
    return full_season_data



season_data_with_features = get_season_data_with_features(2025, 12)
season_data_with_features.to_csv("2025SeasonData.csv", index=False)
