import threading
import queue
import sys
import time
import random
import mne
from datetime import datetime
import os
import pandas as pd
import tkinter as tk
import csv

# EEG CONFIGURATION
unicorn_electrodes = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "M1", "M2"]
# unicorn_electrodes = ["Front, middle", "Front, left", "Center", "Front, right", "middle, a little below", "middle, Left","Behind, middle", "middle, right"]
ch_types = ['eeg'] * len(unicorn_electrodes)
samplingRate = 250
streamName = 'UN-2023.06.33'
output_files = "eeg_data_inheritance/"
raw_data_folder_name = "Raw/"

# available_letters = [chr(i) for i in range(65, 91)]  # A-Z
available_letters = [chr(i) for i in range(0x05D0, 0x05EA)]  # Hebrew letters א-ת
random.shuffle(available_letters)

################################################################
#                      TASK TEST CLASS
################################################################
class TaskTest:
    """
    A unified class for running either:
      - "finger" tapping tasks, or
      - "arithmetic" tasks (3-digit addition) with spacer interaction,
    plus rests in between.

    The test can be used either standalone or as part of EEG collection.
    """

    def __init__(self, test_type="nback", cycles=6, rest_duration=30, task_duration=30
                 ):
        """
        Parameters
        ----------
        test_type : str
            "finger", "arithmetic", "word" or "nback".
        cycles : int
            Number of task+rest cycles.
        rest_duration : int
            Duration (seconds) of rest periods.
        task_duration : int
            Duration (seconds) of the task period.
        """
        self.test_type = test_type
        self.cycles = cycles
        self.rest_duration = rest_duration
        self.task_duration = task_duration

        self.root = tk.Tk()
        self.root.title("Task Test")
        self.root.geometry("800x400")
        self.root.configure(bg="black")

        self.label = tk.Label(self.root, text="", font=("Arial", 50, "bold"),
                              fg="white", bg="black")
        self.label.pack(expand=True)

        self.timer_label = tk.Label(self.root, text="", font=("Arial", 30, "bold"),
                                    fg="yellow", bg="black")
        self.timer_label.pack()

        self.instruction_label = tk.Label(self.root, text="", font=("Arial", 20),
                                          fg="cyan", bg="black")
        self.instruction_label.pack()

        # New feedback label for n-back task
        self.feedback_label = tk.Label(self.root, text="", font=("Arial", 24, "bold"),
                                       fg="white", bg="black")
        self.feedback_label.pack()

        self.timestamps = []
        self.space_pressed = False
        self.task_active = False

        # New variables for n-back task
        self.user_response = None
        self.response_received = False

        # Bind key events
        self.root.bind('<KeyPress-space>', self.on_space_press)
        # self.root.bind('<KeyPress-y>', self.on_y_press)
        # self.root.bind('<KeyPress-n>', self.on_n_press)
        # self.root.bind('<KeyPress-Y>', self.on_y_press)  # Also handle uppercase
        # self.root.bind('<KeyPress-N>', self.on_n_press)  # Also handle uppercase
        # self.root.focus_set()  # Make sure the window can receive key events

    def on_space_press(self, event):
        """Handle spacer press during arithmetic task"""
        if self.task_active:
            self.space_pressed = True

    # def on_y_press(self, event):
    #     """Handle Y key press during n-back task"""
    #     if self.task_active:
    #         self.user_response = True
    #         self.response_received = True
    #
    # def on_n_press(self, event):
    #     """Handle N key press during n-back task"""
    #     if self.task_active:
    #         self.user_response = False
    #         self.response_received = True

    def log_section(self, section_name: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.timestamps.append((section_name, timestamp))
        print(f"{section_name} started at: {timestamp}")

    def display_instruction(self, text, duration_s: int):
        self.label.config(text=text)
        for i in range(duration_s, 0, -1):
            self.timer_label.config(text=f"{i}")
            self.root.update()
            time.sleep(1)
        self.timer_label.config(text="")

    def make_arithmetic_text(self):
        a = random.randint(100, 999)
        b = random.randint(10, 99)
        return f"{a} - {b}?"

    def get_random_letter(self):
        """Generate a random letter from A-Z"""
        return chr(random.randint(65, 90))  # ASCII A-Z

    def run_word_task(self, task_duration,letter_index):
        """
        Run word task that presents random letters throughout the task duration.
        Each letter will only be shown once during the entire task run.
        """
        self.task_active = True
        start_time = time.time()
        # letter_count = 0

        # Create a list of all available letters and shuffle it
        # available_letters = [chr(i) for i in range(65, 91)]  # A-Z
        # random.shuffle(available_letters)
        # letter_index = 0

        while time.time() - start_time < task_duration:
            # Check if we've run out of letters
            # if letter_index >= len(available_letters):
            #     # If we've used all letters, reshuffle and start over
            #     random.shuffle(available_letters)
            #     letter_index = 0
            #     print("All letters used, reshuffling for new cycle...")

            # Get the next letter from our shuffled list
            letter = available_letters[letter_index]
            # letter_index += 1

            self.label.config(text=letter)
            # letter_count += 1

            # Display instruction without logging to timestamps
            self.instruction_label.config(text=f"Please think of words that start with the letter: {letter}")

            # Update timer
            remaining_time = task_duration - (time.time() - start_time)
            self.timer_label.config(text=f"Time: {remaining_time:.1f}s")

            # Show letter for 60 seconds or until task ends
            letter_start = time.time()
            while time.time() - letter_start < 60 and time.time() - start_time < task_duration:
                remaining_time = task_duration - (time.time() - start_time)
                self.timer_label.config(text=f"Time: {remaining_time:.1f}s")
                self.root.update()
                time.sleep(0.1)

        self.task_active = False
        self.timer_label.config(text="")
        self.instruction_label.config(text="")  # Clear instruction when done

    def run_nback_task(self, task_duration):
        """
        Modified N-back task that shows letters for 0.5 seconds with blank intervals.
        No interaction or feedback - just continuous letter presentation.
        """
        # Predefined letter array
        #letter_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        #                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        letter_array = ['A', 'B', 'C']
        time_between_update_time=0.05
        time_for_letter=1
        time_for_blank=1
        self.task_active = True
        start_time = time.time()
        letter_count = 0

        # Clear instruction label for cleaner display
        self.instruction_label.config(text="")
        self.feedback_label.config(text="")

        while time.time() - start_time < task_duration:
            # Select random letter from array
            letter = random.choice(letter_array)
            letter_count += 1

            # Show the letter for 0.5 seconds
            self.label.config(text=letter)
            letter_start = time.time()
            while time.time() - letter_start < time_for_letter and time.time() - start_time < task_duration:
                remaining_time = task_duration - (time.time() - start_time)
                self.timer_label.config(text=f"Time: {remaining_time:.1f}s")
                self.root.update()
                time.sleep(time_between_update_time)  # Small update interval for smooth timer

            # Check if time is up before showing blank
            if time.time() - start_time >= task_duration:
                break

            # Show blank screen (just timer) for 0.5 seconds
            self.label.config(text="")
            blank_start = time.time()
            while time.time() - blank_start < time_for_blank and time.time() - start_time < task_duration:
                remaining_time = task_duration - (time.time() - start_time)
                self.timer_label.config(text=f"Time: {remaining_time:.1f}s")
                self.root.update()
                time.sleep(time_between_update_time)  # Small update interval for smooth timer

        self.task_active = False
        self.label.config(text="")
        self.timer_label.config(text="")
        # print(f"N-back task completed. Total letters shown: {letter_count}")

    def run_arithmetic_task(self, task_duration):
        """
        Run interactive arithmetic task that presents exercises one by one,
        waits for spacer press, then shows next exercise until time runs out.
        """
        self.task_active = True
        start_time = time.time()
        exercise_count = 0

        self.instruction_label.config(text="Press SPACE to get next exercise")

        while time.time() - start_time < task_duration:
            # Generate and display new exercise
            exercise = self.make_arithmetic_text()
            self.label.config(text=exercise)
            exercise_count += 1

            # # Log this exercise
            # self.log_section(f"ARITHMETIC_EXERCISE_{exercise_count}")

            # Update timer
            remaining_time = task_duration - (time.time() - start_time)
            self.timer_label.config(text=f"Time: {remaining_time:.1f}s")

            # Reset space flag and wait for spacer press or timeout
            self.space_pressed = False

            while not self.space_pressed and (time.time() - start_time < task_duration):
                remaining_time = task_duration - (time.time() - start_time)
                self.timer_label.config(text=f"Time: {remaining_time:.1f}s")
                self.root.update()
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage

                # Check if time is up
                if time.time() - start_time >= task_duration:
                    break

        self.task_active = False
        self.instruction_label.config(text="")
        self.timer_label.config(text="")
        print(f"Arithmetic task completed. Total exercises: {exercise_count}")

    def run_test(self):
        print("Starting Task Test...")
        self.log_section("INITIAL_REST")
        self.display_instruction("+\nINITIAL REST", 3)
        #self.log_section("INITIAL_REST")
        #self.display_instruction("+\nINITIAL REST", self.rest_duration)
        letter_index = 0
        for cycle in range(self.cycles):
            if self.test_type.lower() == "finger":
                instruction = "TAP INDEX FINGER"
                self.log_section("TASK_START")
                self.display_instruction(instruction, self.task_duration)

            elif self.test_type.lower() == "arithmetic":
                self.log_section("TASK_START")
                self.run_arithmetic_task(self.task_duration)

            elif self.test_type.lower() == "word":
                self.log_section("TASK_START")
                self.run_word_task(self.task_duration,letter_index)
                letter_index = letter_index+1

            elif self.test_type.lower() == "nback":
                self.log_section("TASK_START")
                self.run_nback_task(self.task_duration)

            self.log_section("REST_START")
            self.display_instruction("+\nREST", self.rest_duration)

        # self.log_section("FINAL_REST")
        # self.display_instruction("REST", self.rest_duration)

        self.log_section("TEST_COMPLETE")
        self.display_instruction("Test Complete!", 3)
        self.root.quit()
        self.root.destroy()

    def save_timestamps(self, path, filename="timestamps.csv"):
        """
        Saves the collected timestamps to a CSV file at the specified file path.
        """
        if path is None:
            path = filename
        elif os.path.isdir(path):
            path = os.path.join(path, filename)

        # If no directory is provided, use current directory
        dir_path = os.path.dirname(path)
        if not dir_path:
            dir_path = "."
        os.makedirs(dir_path, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "StartTimestamp"])
            for section, timestamp in self.timestamps:
                writer.writerow([section, timestamp])

        print(f"Timestamps saved to {path}")

    def start(self):
        self.root.after(100, self.run_test)
        self.root.mainloop()
        # Don't save timestamps here anymore - they will be saved by the App class


################################################################
#             EEG Data Acquisition Logic
################################################################
def sendReal(keepReading, keepPulling, dataQueue, terminate_on_no_eeg=True):
    from pylsl import StreamInlet, resolve_byprop, local_clock
    stream_info = resolve_byprop('name', streamName, 1, 5)
    if len(stream_info) == 0:
        keepReading.clear()
        print("No EEG stream found.")
        if terminate_on_no_eeg:
            print("Terminating due to no EEG connection.")
            sys.exit("No EEG stream found.")
        else:
            print("Continuing without EEG data collection.")
            return False  # Return False to indicate no EEG connection

    inlet = StreamInlet(stream_info[0])
    print("EEG Stream resolved.")

    while keepReading.is_set():
        data, _ = inlet.pull_sample(timeout=0.1)
        if data is None:
            continue
        timestamp = local_clock()
        if keepPulling.is_set() and len(data) >= len(unicorn_electrodes):
            dataQueue.put([timestamp, data[:len(unicorn_electrodes)]])

    return True  # Return True to indicate successful EEG connection


def createMNERawObject(dataQueue):
    listOfData = []
    listOfTimeStamp = []
    while not dataQueue.empty():
        currDataSample = dataQueue.get()
        listOfTimeStamp.append(currDataSample[0])
        listOfData.append(currDataSample[1])
    if len(listOfData) == 0:
        raise Exception("No data found in queue.")
    dataOutputDF = pd.DataFrame(listOfData).T
    info = mne.create_info(ch_names=unicorn_electrodes, sfreq=samplingRate, ch_types=ch_types)
    raw = mne.io.RawArray(dataOutputDF.values, info)
    return raw


def saveData(exp_path, raw):
    date = datetime.now().strftime("%d_%m_%Yat%I_%M_%S_%p")
    file_path = os.path.join(exp_path, raw_data_folder_name)
    os.makedirs(file_path, exist_ok=True)
    raw.save(os.path.join(file_path, f"data_of_{date}.fif"), overwrite=True)
    df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)
    df.insert(0, "Time (s)", raw.times)
    df.to_csv(os.path.join(file_path, f"data_of_{date}.csv"), index=False)


################################################################
#                     APP CLASS
################################################################
class App:
    """
    App has two main modes:
      1) run_eeg = True -> Start EEG acquisition & run the test.
      2) run_eeg = False -> Just run the test alone, no EEG reading.

    The test type can be "finger", "arithmetic", "word", or "nback".

    Parameters:
    - run_eeg: bool, whether to attempt EEG data collection
    - test_type: str, type of task to run
    - terminate_on_no_eeg: bool, whether to terminate if no EEG connection found
    """

    def __init__(self, run_eeg=True, test_type="finger", terminate_on_no_eeg=True):
        self.run_eeg = run_eeg
        self.test_type = test_type
        self.terminate_on_no_eeg = terminate_on_no_eeg
        self.keepReading = threading.Event()
        self.keepReading.set()
        self.keepPulling = threading.Event()
        self.keepPulling.clear()
        self.dataQueue = queue.Queue()
        self.eeg_connected = False

    def run_experiment(self):
        exp_path = os.path.join(output_files, "EXP_" + datetime.now().strftime("%d_%m_%Y_%H%M%S") + "/")
        raw_folder_path = os.path.join(exp_path, raw_data_folder_name)
        os.makedirs(raw_folder_path, exist_ok=True)

        if self.run_eeg:
            print("Attempting to connect to EEG...")

            # Create a result container to get the connection status
            eeg_result = [None]

            def eeg_thread_wrapper():
                result = sendReal(self.keepReading, self.keepPulling, self.dataQueue, self.terminate_on_no_eeg)
                eeg_result[0] = result

            eeg_thread = threading.Thread(target=eeg_thread_wrapper, daemon=True)
            eeg_thread.start()

            # Wait a moment to see if EEG connection is established
            time.sleep(2)

            if eeg_result[0] is False:
                print("No EEG connection found, but continuing with task only...")
                self.eeg_connected = False
                self.run_eeg = False  # Disable EEG for this run
            elif eeg_result[0] is True:
                print("EEG connection established successfully.")
                self.eeg_connected = True
                self.keepPulling.set()
            else:
                # EEG thread is still trying to connect, assume it will work
                print("EEG connection in progress...")
                self.eeg_connected = True
                self.keepPulling.set()

        # Run the task regardless of EEG status
        task_test = TaskTest(test_type=self.test_type)
        task_test.start()

        # Save timestamps to the same Raw folder
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"{self.test_type}_timestamps_{date_str}.csv"
        timestamps_path = os.path.join(raw_folder_path, csv_name)
        task_test.save_timestamps(timestamps_path)

        # Handle EEG data saving only if we actually collected data
        if self.run_eeg and self.eeg_connected:
            self.keepPulling.clear()
            self.keepReading.clear()

            # Wait a moment for threads to stop
            time.sleep(1)

            try:
                raw_obj = createMNERawObject(self.dataQueue)
                saveData(exp_path, raw_obj)
                print("EEG experiment complete. All files saved in Raw folder.")
            except Exception as e:
                print(f"Error saving EEG data: {e}")
                print("Task completed, but EEG data could not be saved.")
        else:
            if self.run_eeg and not self.eeg_connected:
                print(
                    "Task completed. No EEG data was collected due to connection issues. Timestamps saved in Raw folder.")
            else:
                print("Task completed. No EEG recording was requested. Timestamps saved in Raw folder.")


################################################################
#                   MAIN
################################################################
if __name__ == "__main__":
    # Example usage: Run with EEG, terminate if no connection
    app = App(run_eeg=True, test_type="word", terminate_on_no_eeg=False)
    app.run_experiment()
    # The test type can be "finger", "arithmetic", "word", or "nback".  