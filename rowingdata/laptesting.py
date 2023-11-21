import pandas as pd

def summarize_rowing_data(df):
    # Ensure the DataFrame is sorted by time
    df = df.sort_values(by='TimeStamp (sec)')

    # Convert numeric columns to numeric type, handling non-numeric values
    numeric_cols = [
        'TimeStamp (sec)',
        ' Horizontal (meters)',
        ' Cadence (stokes/min)',
        ' HRCur (bpm)',
        ' Stroke500mPace (sec/500m)',
        ' Power (watts)',
        ' DriveLength (meters)',
        ' StrokeDistance (meters)',
        ' DriveTime (ms)',
        ' StrokeRecoveryTime (ms)',
        ' AverageDriveForce (lbs)',
        ' PeakDriveForce (lbs)',
        ' lapIdx',
        ' ElapsedTime (sec)',
        ' WorkoutState'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Initialize variables to store summarized data
    lap_summaries = []

    # Initialize variables to track the start time and data for each lap
    lap_start_time = None
    lap_end_time = None
    lap_cadences = []
    lap_paces = []  # Added lap_paces to store paces for each lap
    lap_distances = []
    lap_powers = []
    lap_heart_rates = []
    lap_id = None

    # Use a mask to filter rows where the workout_state is work strokes
    work_strokes_mask = df[' WorkoutState'].isin([1, 4, 5, 6, 7, 8, 9])

    # Iterate through rows in the DataFrame
    for index, row in df.iterrows():
        if work_strokes_mask[index]:
            # If the lap has started or the lap ID changes, append cadence, pace, and heart rate to lap lists
            if lap_start_time is not None and row[' lapIdx'] == lap_id:
                lap_cadences.append(row[' Cadence (stokes/min)'])
                lap_paces.append(row[' Stroke500mPace (sec/500m)'])
                lap_distances.append(row[' StrokeDistance (meters)'])
                lap_powers.append(row[' Power (watts)'])
                lap_heart_rates.append(row[' HRCur (bpm)'])
                lap_end_time = row['TimeStamp (sec)']
            else:
                # If the lap has not started or the lap ID changes, initialize lap-related variables
                if lap_start_time is not None:
                    # If a lap has ended, calculate lap duration and append averages to summary list
                    lap_duration = lap_end_time - lap_start_time
                    avg_cadence = sum(lap_cadences) / len(lap_cadences)
                    avg_pace = sum(lap_paces) / len(lap_paces)
                    avg_speed = 500 / avg_pace  # Speed is derived from the reciprocal of pace
                    total_distance_per_lap = sum(lap_distances)  # Improved calculation of total distance per lap
                    avg_power = sum(lap_powers) / len(lap_powers)
                    max_heart_rate = max(lap_heart_rates)
                    avg_heart_rate = sum(lap_heart_rates) / len(lap_heart_rates)
                    total_strokes_in_lap = len(lap_cadences)
                    distance_per_stroke = total_distance_per_lap / total_strokes_in_lap

                    lap_summaries.append({
                        'lap_id': lap_id,
                        'lap_duration': lap_duration,
                        'avg_cadence': avg_cadence,
                        'avg_pace': avg_pace,
                        'avg_speed': avg_speed,
                        'total_distance_per_lap': total_distance_per_lap,
                        'avg_power': avg_power,
                        'max_heart_rate': max_heart_rate,
                        'avg_heart_rate': avg_heart_rate,
                        'total_strokes_in_lap': total_strokes_in_lap,
                        'distance_per_stroke': distance_per_stroke
                    })

                # Reset lap-related variables
                lap_start_time = row['TimeStamp (sec)']
                lap_end_time = row['TimeStamp (sec)']
                lap_cadences = [row[' Cadence (stokes/min)']]
                lap_paces = [row[' Stroke500mPace (sec/500m)']]
                lap_distances = [row[' StrokeDistance (meters)']]
                lap_powers = [row[' Power (watts)']]
                lap_heart_rates = [row[' HRCur (bpm)']]
                lap_id = row[' lapIdx']

    # If the last lap is work stroke, add it to the summary
    if lap_start_time is not None:
        lap_duration = lap_end_time - lap_start_time
        avg_cadence = sum(lap_cadences) / len(lap_cadences)
        avg_pace = sum(lap_paces) / len(lap_paces)
        avg_speed = 500 / avg_pace  # Speed is derived from the reciprocal of pace
        total_distance_per_lap = sum(lap_distances)  # Improved calculation of total distance per lap
        avg_power = sum(lap_powers) / len(lap_powers)
        max_heart_rate = max(lap_heart_rates)
        avg_heart_rate = sum(lap_heart_rates) / len(lap_heart_rates)
        total_strokes_in_lap = len(lap_cadences)
        distance_per_stroke = total_distance_per_lap / total_strokes_in_lap

        lap_summaries.append({
            'lap_id': lap_id,
            'lap_duration': lap_duration,
            'avg_cadence': avg_cadence,
            'avg_pace': avg_pace,
            'avg_speed': avg_speed,
            'total_distance_per_lap': total_distance_per_lap,
            'avg_power': avg_power,
            'max_heart_rate': max_heart_rate,
            'avg_heart_rate': avg_heart_rate,
            'total_strokes_in_lap': total_strokes_in_lap,
            'distance_per_stroke': distance_per_stroke
        })

    # Create a new DataFrame with the summarized data
    summary_df = pd.DataFrame(lap_summaries)

    return summary_df

import pandas as pd


import pandas as pd

def summarize_entire_workout(df):
    # Ensure the DataFrame is sorted by time
    df = df.sort_values(by='TimeStamp (sec)')

    # Convert numeric columns to numeric type, handling non-numeric values
    numeric_cols = [
        'TimeStamp (sec)',
        ' Horizontal (meters)',
        ' Cadence (stokes/min)',
        ' HRCur (bpm)',
        ' Stroke500mPace (sec/500m)',
        ' Power (watts)',
        ' DriveLength (meters)',
        ' StrokeDistance (meters)',
        ' DriveTime (ms)',
        ' StrokeRecoveryTime (ms)',
        ' AverageDriveForce (lbs)',
        ' PeakDriveForce (lbs)',
        ' lapIdx',
        ' ElapsedTime (sec)',
        ' WorkoutState'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Use a mask to filter rows where the workout_state is work strokes or resting strokes
    work_strokes_mask = df[' WorkoutState'].isin([1, 4, 5, 6, 7, 8, 9])
    rest_strokes_mask = df[' WorkoutState'] == 3

    # Filter rows for work strokes, resting strokes, and entire workout
    work_intervals_df = df[work_strokes_mask]
    rest_intervals_df = df[rest_strokes_mask]

    # Calculate total distance for work intervals and resting intervals
    total_distance_work_intervals = (
        work_intervals_df.groupby(' lapIdx')[' Horizontal (meters)'].max().diff().fillna(
            work_intervals_df.groupby(' lapIdx')[' Horizontal (meters)'].max()
        ).sum()
    )
    total_distance_rest_intervals = (
        rest_intervals_df.groupby(' lapIdx')[' Horizontal (meters)'].max().diff().fillna(
            rest_intervals_df.groupby(' lapIdx')[' Horizontal (meters)'].max()
        ).sum()
    )

    # Calculate total distance for the entire workout
    total_distance_workout = (
        df.groupby(' lapIdx')[' Horizontal (meters)'].max().diff().fillna(
            df.groupby(' lapIdx')[' Horizontal (meters)'].max()
        ).sum()
    )

    # Get the maximum total time among all laps
    total_time = df.groupby(' lapIdx')[' ElapsedTime (sec)'].max().max()

    # Calculate averages for the entire workout
    avg_cadence_workout = df[' Cadence (stokes/min)'].mean()
    avg_pace_workout = df[' Stroke500mPace (sec/500m)'].mean()
    avg_speed_workout = 500 / avg_pace_workout
    avg_power_workout = df[' Power (watts)'].mean()
    avg_heart_rate_workout = df[' HRCur (bpm)'].mean()

    # Calculate averages for work intervals
    avg_cadence_work = work_intervals_df[' Cadence (stokes/min)'].mean()
    avg_pace_work = work_intervals_df[' Stroke500mPace (sec/500m)'].mean()
    avg_speed_work = 500 / avg_pace_work
    avg_power_work = work_intervals_df[' Power (watts)'].mean()
    avg_heart_rate_work = work_intervals_df[' HRCur (bpm)'].mean()

    # Calculate averages for resting intervals
    avg_cadence_rest = rest_intervals_df[' Cadence (stokes/min)'].mean()
    avg_pace_rest = rest_intervals_df[' Stroke500mPace (sec/500m)'].mean()
    avg_speed_rest = 500 / avg_pace_rest
    avg_power_rest = rest_intervals_df[' Power (watts)'].mean()
    avg_heart_rate_rest = rest_intervals_df[' HRCur (bpm)'].mean()

    # Create a summary dictionary
    summary_dict = {
        'total_distance_workout': total_distance_workout,
        'total_distance_work_intervals': total_distance_work_intervals,
        'total_distance_rest_intervals': total_distance_rest_intervals,
        'total_time': total_time,
        'avg_cadence_workout': avg_cadence_workout,
        'avg_pace_workout': avg_pace_workout,
        'avg_speed_workout': avg_speed_workout,
        'avg_power_workout': avg_power_workout,
        'avg_heart_rate_workout': avg_heart_rate_workout,
        'avg_cadence_work': avg_cadence_work,
        'avg_pace_work': avg_pace_work,
        'avg_speed_work': avg_speed_work,
        'avg_power_work': avg_power_work,
        'avg_heart_rate_work': avg_heart_rate_work,
        'avg_cadence_rest': avg_cadence_rest,
        'avg_pace_rest': avg_pace_rest,
        'avg_speed_rest': avg_speed_rest,
        'avg_power_rest': avg_power_rest,
        'avg_heart_rate_rest': avg_heart_rate_rest
    }

    return summary_dict

def create_workout_summary_string(summary_df, summary_dict):
    # Extract values from the summary_dict
    total_distance_workout = int(summary_dict['total_distance_workout'])
    total_distance_work_intervals = int(summary_dict['total_distance_work_intervals'])
    total_distance_rest_intervals = int(summary_dict['total_distance_rest_intervals'])
    total_time = str(summary_dict['total_time'])
    avg_cadence_workout = str(round(summary_dict['avg_cadence_workout'], 1))
    avg_pace_workout = str(summary_dict['avg_pace_workout'])
    avg_speed_workout = str(round(summary_dict['avg_speed_workout'], 1))
    avg_power_workout = str(round(summary_dict['avg_power_workout'], 1))
    avg_heart_rate_workout = str(round(summary_dict['avg_heart_rate_workout'], 1))
    avg_cadence_work = str(round(summary_dict['avg_cadence_work'], 1))
    avg_pace_work = str(summary_dict['avg_pace_work'])
    avg_speed_work = str(round(summary_dict['avg_speed_work'], 1))
    avg_power_work = str(round(summary_dict['avg_power_work'], 1))
    avg_heart_rate_work = str(round(summary_dict['avg_heart_rate_work'], 1))
    avg_cadence_rest = str(round(summary_dict['avg_cadence_rest'], 1))
    avg_pace_rest = str(summary_dict['avg_pace_rest'])
    avg_speed_rest = str(round(summary_dict['avg_speed_rest'], 1))
    avg_power_rest = str(round(summary_dict['avg_power_rest'], 1))
    avg_heart_rate_rest = str(round(summary_dict['avg_heart_rate_rest'], 1))

    # Create the workout summary string
    workout_summary_string = (
        f"Workout Summary - testdata.csv\n"
        "--|Total|-Total----|--Avg--|-Avg-|Avg-|-Avg-|-Max-|-Avg\n"
        f"--|{total_distance_workout:05d}|{total_time}|{avg_pace_workout}|{avg_power_workout}|"
        f"{avg_cadence_workout}|{avg_heart_rate_workout}|XXX.X|{avg_heart_rate_workout}\n"
        f"W-|{total_distance_work_intervals:05d}|{total_time}|{avg_pace_work}|{avg_power_work}|"
        f"{avg_cadence_work}|{avg_heart_rate_work}|XXX.X|{avg_heart_rate_work}\n"
        f"R-|{total_distance_rest_intervals:05d}|00:00.0|00:00.0|000.0|00.0|000.0|{avg_heart_rate_rest}|00.0\n"
        "Workout Details\n"
        "#-|SDist|-Split-|-SPace-|-Pwr-|SPM-|AvgHR|MaxHR|DPS-\n"
    )

    # Iterate over rows in the summary DataFrame to create details
    for _, row in summary_df.iterrows():
        lap_distance = int(row['total_distance_per_lap'])
        split_time = str(row['lap_duration'])
        split_pace = str(row['avg_pace'])
        split_power = str(row['avg_power'])
        split_cadence = str(row['avg_cadence'])
        avg_heart_rate_split = str(row['avg_heart_rate'])
        max_heart_rate_split = str(row['max_heart_rate'])
        dps = str(row['distance_per_stroke'])
        lap_id = int(row['lap_id'])

        workout_summary_string += (
            f"{lap_id:02d}|{lap_distance:05d}|{split_time}|{split_pace}|{split_power}|"
            f"{split_cadence}|{avg_heart_rate_split}|{max_heart_rate_split}|{dps}\n"
        )

    return workout_summary_string

# Example usage:
# Assuming your DataFrame is named 'rowing_data_df'
#summary_entire_workout = summarize_entire_workout(rowing_data_df)
#print(summary_entire_workout)
