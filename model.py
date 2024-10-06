# model.py

from pathlib import Path
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from scipy import signal
from matplotlib import cm
import numpy as np
from sys import argv
import csv
import datetime
import os

def main():
    if len(argv) != 2:
        print(f'Usage: {argv[0]} <path to directory with mseed files>')
        return

    mseed_directory = Path(argv[1])
    if not mseed_directory.is_dir():
        print(f'Error: {mseed_directory} is not a valid directory.')
        return

    mseed_pathlist = list(mseed_directory.glob('*.mseed'))

    if not mseed_pathlist:
        print(f'No .mseed files found in {mseed_directory}.')
        return

    catalogue = []

    evid_counter = 1 

    DEFAULT_MQ_TYPE = 'impac_mq'

    plots_directory = Path('detections_plots')
    plots_directory.mkdir(exist_ok=True)
    for mseed_path in mseed_pathlist:
        freq_min = 0.5
        freq_max = 1.0
        
        print(f'Processing file: {mseed_path.name}')

        try:
            st = read(str(mseed_path))
        except Exception as e:
            print(f'Error reading {mseed_path.name}: {e}, skipping.')
            continue

        if not st:
            print(f'No traces found in {mseed_path.name}, skipping.')
            continue

        trace = st.traces[0]
        start_time = trace.stats.starttime 

        st_filt = st.copy()
        try:
            st_filt.filter('bandpass', freqmin=freq_min, freqmax=freq_max)
        except Exception as e:
            print(f'Error filtering {mseed_path.name}: {e}, skipping.')
            continue

        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data

        try:
            spect_freqs, spect_times, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
        except Exception as e:
            print(f'Error computing spectrogram for {mseed_path.name}: {e}, skipping.')
            continue

        print(f"Spectrogram power range: min = {np.min(sxx)}, max = {np.max(sxx)}")

        baseline_end_idx = int(0.1 * len(spect_times)) 
        baseline_noise = np.mean(sxx[:, :baseline_end_idx]) 
        print(f"Baseline noise level: {baseline_noise}")

        power_threshold = 0.01 * np.max(sxx)  
        noise_threshold = baseline_noise 
        print(f"Event Start Threshold (1% of max): {power_threshold}")
        print(f"Noise Threshold (Baseline): {noise_threshold}")

        mean_power = np.mean(sxx, axis=0)
        detection_indices = np.where(mean_power > power_threshold)[0]

        predicted_times = []
        predicted_end_times = []
        MIN_DURATION = 250  

        i = 0
        while i < len(detection_indices):
            predicted_time_idx = detection_indices[i]
            predicted_time = spect_times[predicted_time_idx]

            window_size = 10  
            mean_power_after_start = mean_power[predicted_time_idx:]

            for j in range(len(mean_power_after_start) - window_size):
                window_avg = np.mean(mean_power_after_start[j:j+window_size])
                if window_avg < noise_threshold:
                    predicted_end_time_idx = predicted_time_idx + j
                    predicted_end_time = spect_times[predicted_end_time_idx]
                    break
            else:
                predicted_end_time = None

            if predicted_time is not None and predicted_end_time is not None:
                duration = predicted_end_time - predicted_time
                if duration >= MIN_DURATION:  
                    predicted_times.append(predicted_time)
                    predicted_end_times.append(predicted_end_time)
                    print(f"Valid detection found from {predicted_time} to {predicted_end_time} (duration: {duration}s)")

                    time_rel_sec = predicted_time  
                    time_abs = start_time + time_rel_sec 

                    time_abs_str = time_abs.strftime("%Y-%m-%d %H:%M:%S.%f")

                    evid_str = f"evid{evid_counter:05d}"

                    filename_with_evid = f"{mseed_path.stem}_evid{evid_counter:05d}.png"  

                    catalogue.append({
                        'filename': filename_with_evid,
                        'time_abs': time_abs_str,
                        'time_rel': f"{time_rel_sec:.1f}",
                        'evid': evid_str,
                        'mq_type': DEFAULT_MQ_TYPE
                    })

                    evid_counter += 1  
                else:
                    print(f"Duration too short: {duration}s, skipping detection")

            if predicted_end_time is not None:
                i = np.searchsorted(detection_indices, predicted_end_time_idx + 1)
            else:
                i += 1

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        start_lines = []
        end_lines = []

        def clear_old_lines():
            for line in start_lines + end_lines:
                if line in ax.lines:
                    ax.lines.remove(line)
            start_lines.clear()
            end_lines.clear()

        def plot_detections():
            clear_old_lines()

            ax.plot(tr_times_filt, tr_data_filt, label='Seismic Signal')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title(f'Seismic Signal with Detections - {mseed_path.name}')

            if predicted_times and predicted_end_times:
                latest_predicted_time = predicted_times[-1]
                latest_predicted_end_time = predicted_end_times[-1]

                if min(tr_times_filt) <= latest_predicted_time <= max(tr_times_filt):
                    start_line = ax.axvline(x=latest_predicted_time, color='green', linestyle='--', label='Predicted Start Detection')
                    start_lines.append(start_line)
                if min(tr_times_filt) <= latest_predicted_end_time <= max(tr_times_filt):
                    end_line = ax.axvline(x=latest_predicted_end_time, color='blue', linestyle='--', label='Predicted End Detection')
                    end_lines.append(end_line)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')

            ax.set_xlim([min(tr_times_filt), max(tr_times_filt)])
            ax.set_xlabel('Time (s)')

            pcm = ax2.pcolormesh(spect_times, spect_freqs, sxx, cmap=cm.jet, shading='gouraud')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Spectrogram')

            if predicted_times and predicted_end_times:
                latest_predicted_time = predicted_times[-1]
                latest_predicted_end_time = predicted_end_times[-1]

                if min(spect_times) <= latest_predicted_time <= max(spect_times):
                    ax2.axvline(x=latest_predicted_time, color='green', linestyle='--')
                if min(spect_times) <= latest_predicted_end_time <= max(spect_times):
                    ax2.axvline(x=latest_predicted_end_time, color='blue', linestyle='--')

            cbar = fig.colorbar(pcm, ax=ax2, orientation='horizontal', pad=0.1)
            cbar.set_label('Power ((m/s)^2/Hz)', fontweight='bold')

            plt.tight_layout()

            plot_filename = plots_directory / f"{mseed_path.stem}.png"
            try:
                plt.savefig(plot_filename)
                print(f"Plot saved as {plot_filename}")
            except Exception as e:
                print(f"Error saving plot for {mseed_path.name}: {e}")

            plt.close(fig)

        plot_detections()

    def write_catalogue(catalogue, output_path='catalogue.csv'):
        if not catalogue:
            print("No detections to write to catalogue.")
            return

        headers = ['filename', 'time_abs(%Y-%m-%d %H:%M:%S.%f)', 'time_rel(sec)', 'evid', 'mq_type']

        try:
            with open(output_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for event in catalogue:
                    writer.writerow({
                        'filename': event['filename'],
                        'time_abs(%Y-%m-%d %H:%M:%S.%f)': event['time_abs'],
                        'time_rel(sec)': event['time_rel'],
                        'evid': event['evid'],
                        'mq_type': event['mq_type']
                    })
            print(f"Catalogue successfully written to {output_path}")
        except Exception as e:
            print(f"Error writing catalogue to {output_path}: {e}")

    main()

    write_catalogue(catalogue)

if __name__ == "__main__":
    main()
