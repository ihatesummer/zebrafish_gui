import graph_generator as gg

from kivy.uix.screenmanager import Screen
from os import getcwd, makedirs
from os.path import join, normpath, basename, exists
from numpy import (append, convolve,
                   linspace, loadtxt,
                   ones, roll)
from numpy.core.fromnumeric import nonzero
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from datetime import datetime
from scipy.signal import find_peaks

class Plotter_Window(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vid_name = ""
        self.result_path = join(
            getcwd(), "results", self.vid_name)
        self.data_file = ""
        self.graph_file = ""
        self.eye_selection = ["left", "right"]
        self.axes_selection = {
            "x": "time",
            "y": "angle"}
        self.custom_label = ["", ""]
        self.custom_eye_label = ["Left", "Right"]
        self.custom_grid = [None, None]
        self.custom_colors = ["blue", "green"]
        self.graph_title = ""
        self.wrt_B = True
        self.bNoData = False
        self.x_range = [0, 0]
        self.y_range = [0, 0]
        self.window_size = 1
        self.fps = 1
        self.peak_prominence = 10
        self.peak_margins = [0.5, 0.5]
        self.fft_timeRange = [0, 0]
        self.c_frame_no = None
        self.c_time = None
        self.c_bDetected = None
        self.c_angle_B = None
        self.c_angle_L = None
        self.c_angle_wrtB_L = nonzero
        self.c_angle_R = None
        self.c_angle_wrtB_R = None

    def set_fft_timeRange(self):
        try:
            self.fft_timeRange[0] = float(self.ids.fft_timeRange_from.text)
            self.fft_timeRange[1] = float(self.ids.fft_timeRange_to.text)
        except:
            print("ERROR: Please enter float for fft range.")
        print("Update - fft range: ", self.fft_timeRange)

    def set_custom_colors(self):
        try:
            self.custom_colors[0] = self.ids.graph_color_left.text
            self.custom_colors[1] = self.ids.graph_color_right.text
        except:
            print("ERROR: Wrong value for color.")
        print("Update - custom colors: ", self.custom_colors)

    def set_graph_title(self):
        try:
            self.graph_title = self.ids.graph_title.text
        except:
            print("ERROR: wrong value for graph title")
        print("Update - graph title: ", self.graph_title)

    def set_window_size(self):
        try:
            self.window_size = int(self.ids.window_size.text)
            print("Update - window size: ", self.window_size)
        except:
            print("ERROR: Please enter an integer for window size.")

    def set_peak_prominence(self):
        try:
            self.peak_prominence = float(self.ids.prominence.text)
            print("Update - Peak prominence: ", self.peak_prominence)
        except:
            print("ERROR: Please enter a floating point for peak prominence.")

    def set_peak_margins(self, left_or_right):
        try:
            if left_or_right == "left":
                self.peak_margins[0] = float(self.ids.peak_margin_l.text)
            if left_or_right == "right":
                self.peak_margins[1] = float(self.ids.peak_margin_r.text)
            print("Update - Peak margins: ", self.peak_margins)
        except:
            print("ERROR: Please enter floating points for margins.")

    def get_timestamp_namepath(self):
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d-%H_%M_%S.png")
        timestamp_namepath = join(
            self.result_path, date_time)
        return timestamp_namepath

    def show_peaks(self):
        if self.bNoData:
            print("No datasheet available.")
            return None
        self.ids.x_ax_time.active = True
        self.update_axes_selection(self, self.ids.x_ax_time.active, "x", "time")
        self.ids.y_ax_angVel.active = True
        self.update_axes_selection(self, self.ids.y_ax_angVel.active, "y", "angVel")

        x, x_label = self.fetch_x()
        y, y_label, _ = self.fetch_y()
        output_namepath = self.get_timestamp_namepath()
        gg.peak_preview(output_namepath,
                        x, x_label, self.x_range,
                        y, y_label, self.y_range,
                        self.eye_selection,
                        self.custom_grid,
                        self.custom_label,
                        self.custom_eye_label,
                        self.custom_colors,
                        self.graph_title,
                        self.peak_prominence)
        self.graph_file = output_namepath
        self.ids.preview_graph.source = self.graph_file
        self.ids.y_ax_SPV.active = True
        self.update_axes_selection(self, self.ids.y_ax_SPV.active, "y", "spv")

    def moving_average(self, x):
        w = self.window_size
        moving_avg = convolve(x, ones(w), 'valid') / w
        # duplicate the last element
        # in order to fit the original length
        m_avg_extended = append(moving_avg,
                                ones(w-1)*moving_avg[-1])
        return m_avg_extended

    def get_angVel(self, angle_list):
        deltaTime = 1/self.fps
        angVel_list = angle_list.copy()
        for i in range(1, len(angVel_list)):
            angVel_list[i] = \
                (angle_list[i] - angle_list[i-1]) / deltaTime
        angVel_list[0] = angVel_list[1]
        return angVel_list

    def get_nPeak(self, y):
        peaks, _ = find_peaks(-y,
                              self.peak_prominence,
                              distance=self.fps)
        return len(peaks)

    
    def get_slowPhase_idx(self, y):
        peaks, _ = find_peaks(-y,
                              self.peak_prominence,
                              distance=self.fps)
        # Initialize as all True
        idx_slowPhase = ones(len(y), dtype=bool) 
        left_margin = int(self.peak_margins[0]*self.fps)
        right_margin = int(self.peak_margins[1]*self.fps)

        for i in range (-left_margin, right_margin):
            try:
                idx_slowPhase[peaks+i] = False
            except Exception as e:
                print("ERROR setting peak indices"
                      f"at {peaks+i}: {e}" + 
                      "Reason 1. first peak may be too close to index zero." + 
                      "Reason 2. last peak may be too close to the last index")
                
                pass
        return idx_slowPhase

    def preprocess_angles(self, angle):
        angle_processed = self.moving_average(angle)
        angVel_processed = self.get_angVel(
            angle_processed)
        return angle_processed, angVel_processed

    def choose_angle_data(self, angle, angVel, y_selected):
        if y_selected == "angle":
            y = angle
            y_label = "Angle [degree]"
            idx = [None]
        elif y_selected == "angVel":
            y = angVel
            y_label = "Angular velocity [degree/sec]"
            idx = [None]
        elif y_selected == "spv":
            y = angVel
            y_label = "Slow phase velocity [degree/sec]"
            idx = self.get_slowPhase_idx(angVel)
        elif y_selected == "beats":
            y = angVel
            y_label = ""
            idx = [None]
        elif y_selected == "sacc_freq":
            peak_frames, _ = find_peaks(
                -angVel,
                self.peak_prominence,
                distance=self.fps)
            peak_times = self.c_time[peak_frames]
            shifted = roll(self.c_time[peak_frames], 1)
            shifted[0] = 0
            sacc_freq = 1 / (peak_times - shifted)

            y = sacc_freq
            y_label = "Saccade frequency [Hz]"
            idx = peak_frames
        else:
            return None
        return y, y_label, idx

    def get_eyeData_selection(self,
                              eyeData_collection,
                              selection):
        (area, axRatio,
         angle_wrtB, angle) = eyeData_collection
        if selection == "area":
            y = self.moving_average(area)
            return y, "Normalized area [0-1]", [None]
        if selection == "axRatio":
            y = self.moving_average(axRatio)
            return y, "Axes ratio [major:minor]", [None]

        if self.wrt_B == True:
            angle, angVel = self.preprocess_angles(
                angle_wrtB)
            y, y_label, idx = self.choose_angle_data(
                angle, angVel, selection)
        else:
            angle, angVel = self.preprocess_angles(
                angle)
            y, y_label, idx = self.choose_angle_data(
                angle, angVel, selection)
        return y, y_label, idx

    def fetch_x(self):
        if self.axes_selection["x"] == "frame":
            x = self.c_frame_no
            xlabel = "Frame"
        elif self.axes_selection["x"] == "time":
            x = self.c_time
            xlabel = "Time [sec]"
        elif self.axes_selection["x"] == "freq":
            x = None
            xlabel = "Frequency [Hz]"
        else:
            print("ERROR: No x-axis option selected.")
        print(f"{xlabel} fetched for x-axis")
        return x, xlabel

    def fetch_y(self):
        y_list = []
        y_labels = []
        idx_list = []
        y_selected = self.axes_selection["y"]
        if "left" in self.eye_selection:
            left_eyeData_collection = (
                self.c_area_norm_L,
                self.c_ax_ratio_L,
                self.c_angle_wrtB_L,
                self.c_angle_L
            )
            y, y_label, idx = self.get_eyeData_selection(
                left_eyeData_collection,
                y_selected)
            y_list.append(y)
            y_labels.append(y_label)
            idx_list.append(idx)
            
        if "right" in self.eye_selection:
            right_eyeData_collection = (
                self.c_area_norm_R,
                self.c_ax_ratio_R,
                self.c_angle_wrtB_R,
                self.c_angle_R
            )
            y, y_label, idx = self.get_eyeData_selection(
                right_eyeData_collection,
                y_selected)
            y_list.append(y)
            y_labels.append(y_label)
            idx_list.append(idx)
        # two identical labels are appended during 
        # the if/else statements above
        y_label = y_labels[0]
        print(f"{y_label} fetched for y-axis")

        return y_list, y_label, idx_list

    def get_first_available(self, y_part, bDetected_part):
        if bDetected_part[0] == True:
            return y_part[0]
        else:
            # Recursion until available
            return self.get_first_available(
                y_part[1:], bDetected_part[1:]
            )

    def get_last_available(self, y_part, bDetected_part):
        if bDetected_part[-1] == True:
            return y_part[-1]
        else:
            # Recursion until available
            return self.get_last_available(
                y_part[:-1], bDetected_part[:-1])

    def interpolate(self, y):
        idx_last = int(max(self.c_frame_no))
        if self.c_bDetected[0] == False:
            y[0] = self.get_first_available(
                        y[1:], self.c_bDetected[1:])
        if self.c_bDetected[idx_last] == False:
            y[idx_last] = self.get_last_available(
                        y[:-1], self.c_bDetected[:-1])
        for i in range (1, idx_last-1):
            if self.c_bDetected[i] == False:
                if len(y[:i]) == 1:
                    prev = y[:i]
                else:
                    prev = self.get_last_available(
                        y[:i], self.c_bDetected[:i])
                if len(y[i+1:]) == 1:
                    next = y[i+1:]
                else:
                    next = self.get_first_available(
                        y[i+1:], self.c_bDetected[i+1:])
                y[i] = (prev + next) / 2
            else:
                pass
        return y

    def get_bpm(self, angVel):
        # bpm: beats per minute
        nPeak = self.get_nPeak(angVel)
        TimeDuration_sec = len(angVel) / self.fps
        TimeDuration_min = TimeDuration_sec / 60
        bpm = nPeak/TimeDuration_min
        return bpm

    def generate_graph(self):
        if self.bNoData:
            print("No datasheet available.")
            return None
        output_namepath = self.get_timestamp_namepath()

        try:
            x, xlabel = self.fetch_x()
            y, y_label, idx = self.fetch_y()

            if self.axes_selection["y"] == "spv":
                gg.show_spv(output_namepath,
                            self.c_time, xlabel, self.x_range,
                            y, y_label, self.y_range,
                            self.eye_selection,
                            self.custom_grid,
                            self.custom_label,
                            self.custom_eye_label,
                            self.custom_colors,
                            self.graph_title, idx)
                self.graph_file = output_namepath
                self.ids.preview_graph.source = self.graph_file
                return None

            if self.axes_selection["y"] == "beats":
                bpms = []
                for y_arr in y:
                    bpm = self.get_bpm(y_arr)
                    bpms.append(bpm)
                gg.show_bpm(output_namepath, bpms, self.eye_selection)
                self.graph_file = output_namepath
                self.ids.preview_graph.source = self.graph_file
                return None
            
            if self.axes_selection["y"] == "sacc_freq":
                gg.show_sacc_freq(output_namepath,
                            self.c_time, xlabel, self.x_range,
                            y, y_label, self.y_range,
                            self.eye_selection,
                            self.custom_grid,
                            self.custom_label,
                            self.custom_eye_label,
                            self.custom_colors,
                            self.graph_title, idx)
                self.graph_file = output_namepath
                self.ids.preview_graph.source = self.graph_file
                return None         

            if self.axes_selection["x"] == "freq":
                idx_start = int(
                    self.fft_timeRange[0]*self.fps)
                idx_end = int(
                    self.fft_timeRange[1]*self.fps)
                for i, y_arr in enumerate(y):
                    y_arr = y_arr[idx_start:idx_end]
                    # y[i] = abs(rfft(y_arr)) / len(y_arr)
                    _, y[i] = welch(y_arr, self.fps)
                nSamples = len(
                    self.c_time[idx_start:idx_end])
                sample_interval = 1 / self.fps
                # x = rfftfreq(nSamples, sample_interval)
                x, _ = welch(y_arr, self.fps)
                idx_y_unit = y_label.index("[") - 1
                y_label = y_label[:idx_y_unit] + " - Amplitude"

            gg.main(output_namepath,
                    x, xlabel, self.x_range,
                    y, y_label, self.y_range,
                    self.eye_selection,
                    self.custom_grid,
                    self.custom_label,
                    self.custom_eye_label,
                    self.custom_colors,
                    self.graph_title,
                    self.axes_selection["x"])
            self.graph_file = output_namepath
            self.ids.preview_graph.source = self.graph_file
        except:
            print("ERROR: invalid configuration(s).")

    def update_wrtB(self, instance, value):
        self.wrt_B = value
        print("Update - with respect to body: ", self.wrt_B)

    def update_eye_selection(self, instance, value, LR):
        if value == True:
            self.eye_selection.append(LR)
        else:
            self.eye_selection.remove(LR)
        print("Update - eye selection: ", self.eye_selection)

    def set_x_range(self, idx):
        try:
            if idx == "from":
                self.x_range[0] = float(self.ids.x_range_from.text)
            elif idx == "to":
                self.x_range[1] = float(self.ids.x_range_to.text)
            else:
                pass
        except:
            print("ERROR: Invalid input for x-axis range.")
        print("Update - x_range: ", self.x_range)

    def clear_x_range(self, instance, value):
        try:
            if value == True:
                self.x_range = [0, 0]
            else:
                self.x_range[0] = float(self.ids.x_range_from.text)
                self.x_range[1] = float(self.ids.x_range_to.text)
        except:
            print("ERROR: Invalid input for x-axis range.")
        print("Update - x_range: ", self.x_range)

    def set_y_range(self, idx):
        try:
            if idx == "from":
                self.y_range[0] = float(self.ids.y_range_from.text)
            elif idx == "to":
                self.y_range[1] = float(self.ids.y_range_to.text)
            else:
                pass
        except:
            print("ERROR: Invalid input for y-axis range.")
        print("Update - y_range: ", self.y_range)

    def clear_y_range(self, instance, value):
        try:
            if value == True:
                self.y_range = [0, 0]
            else:
                self.y_range[0] = float(self.ids.y_range_from.text)
                self.y_range[1] = float(self.ids.y_range_to.text)
        except:
            print("ERROR: Invalid input for x-axis range.")
        print("Update - y_range: ", self.y_range)

    def set_grid(self, ax):
        if ax == "x":
            try:
                a = int(self.ids.x_grid_from.text)
                b = int(self.ids.x_grid_to.text)
                c = int(self.ids.x_grid_count.text)
                self.custom_grid[0] = linspace(a, b, c)
            except:
                self.custom_grid[0] = None
        elif ax == "y":
            try:
                a = int(self.ids.y_grid_from.text)
                b = int(self.ids.y_grid_to.text)
                c = int(self.ids.y_grid_count.text)
                self.custom_grid[1] = linspace(a, b, c)
            except:
                self.custom_grid[1] = None
        else:
            pass
        print("Update - custom grid: ", self.custom_grid)

    def clear_grid(self, instance, value, ax):
        try:
            if value == True:
                if ax == "x":
                    self.custom_grid[0] = None
                elif ax == "y":
                    self.custom_grid[1] = None
                else:
                    pass
            else:
                if ax == "x":
                    self.set_grid("x")
                elif ax == "y":
                    self.set_grid("y")
                else:
                    pass
        except Exception as e:
            print(f"ERROR: {e}")
        print("Update - custom grid: ", self.custom_grid)

    def set_custom_label(self, ax):
        if ax == "x":
            self.custom_label[0] = self.ids.custom_xlabel.text
        elif ax == "y":
            self.custom_label[1] = self.ids.custom_ylabel.text
        else:
            pass
        print("Update - custom axes label: ", self.custom_label)

    def set_custom_eye_label(self, lr):
        if lr == "left":
            self.custom_eye_label[0] = self.ids.graph_label_left.text
        elif lr == "right":
            self.custom_eye_label[1] = self.ids.graph_label_right.text
        else:
            pass
        print("Update - custom eyes label: ", self.custom_eye_label)


    def clear_custom_label(self, instance, value, ax):
        if value == True:
            if ax == "x":
                self.custom_label[0] = ""
            elif ax == "y":
                self.custom_label[1] = ""
            else:
                pass
        else:
            if ax == "x":
                self.custom_label[0] = self.ids.custom_xlabel.text
            elif ax == "y":
                self.custom_label[1] = self.ids.custom_ylabel.text
            else:
                pass
        print("Update - custom label: ", self.custom_label)

    def update_axes_selection(self, instance, value, axis, choice):
        if value == True:
            self.axes_selection[axis] = choice
        else:
            self.axes_selection[axis] = ""
            # when frequency choice is disabled for x-axis,
            # reset the y-axis choice.
            if choice == "freq":
                self.reset_y_ax_choice()

        if self.axes_selection["y"] == "angVel":
            self.ids.x_ax_frame.active = False
            self.ids.x_ax_frame.disabled = True
        else:
            self.ids.x_ax_frame.disabled = False

        print("Update - axes selection: ",
              self.axes_selection)

    def reset_y_ax_choice(self):
        if self.ids.y_ax_angle.active == True:
            self.axes_selection["y"] = "angle"
        elif self.ids.y_ax_angVel.active == True:
            self.axes_selection["y"] = "angVel"
        elif self.ids.y_ax_area.active == True:
            self.axes_selection["y"] = "area"
        elif self.ids.y_ax_axRatio.active == True:
            self.axes_selection["y"] = "axRatio"
        else:
            self.axes_selection["y"] = ""

    def createFolder(self, dir: str) -> None:
        """
        Create a directory if it does not already exists.
        :param dir: directory name
        """
        try:
            if not exists(dir):
                makedirs(dir)
        except Exception as e:
            print(f"Error: Creating directory {dir} failed: {e}")

    def select_vid(self):
        vid_path = self.ids.filechooser.selection[0]
        self.vid_name = basename(normpath(vid_path))
        self.result_path = join(
            getcwd(), "results", self.vid_name)
        self.ids.vid_selected.text = \
            f"[color=aaff00]{self.vid_name}[/color] selected."
        self.data_file = join(self.result_path, "result.csv")

        default_graph = join(
            self.result_path, ".blank.png")
        if not exists(default_graph):
            self.createFolder(self.result_path)
            gg.generate_blank(default_graph)
        self.graph_file = default_graph
        self.ids.preview_graph.source = self.graph_file
        self.load()
    
    def load(self):
        try:
            data_frame = loadtxt(
                self.data_file,
                delimiter=',')
            # Sort by frame number
            data_frame = data_frame[
                data_frame[:, 0].argsort()]
            # Load each column
            self.c_frame_no = data_frame[:, 0]
            self.c_time = data_frame[:, 1]
            self.c_bDetected = data_frame[:, 2]
            self.c_angle_B = self.interpolate(data_frame[:, 3])
            self.c_angle_L = self.interpolate(data_frame[:, 4])
            self.c_angle_wrtB_L = self.interpolate(data_frame[:, 5])
            self.c_angle_R = self.interpolate(data_frame[:, 6])
            self.c_angle_wrtB_R = self.interpolate(data_frame[:, 7])
            self.c_area_norm_L = self.interpolate(data_frame[:, 8])
            self.c_area_norm_R = self.interpolate(data_frame[:, 9])
            self.c_ax_ratio_L = self.interpolate(data_frame[:, 10])
            self.c_ax_ratio_R = self.interpolate(data_frame[:, 11])
            self.fps = 1 / (self.c_time[1] - self.c_time[0])
            self.bNoData = False
            print(f"{self.vid_name} [{int(self.fps)}FPS] data loaded.")

        except:
            print(f"ERROR: Processing result is not available for {self.vid_name}.")
            self.bNoData = True
