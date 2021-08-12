from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.lang import Builder
from cv2 import imwrite, cvtColor, COLOR_RGB2BGR
from os import getcwd, listdir
from os.path import join, normpath, basename, exists
from json import dump, load
from numpy import (abs, vstack, savetxt, count_nonzero,
                   linspace, loadtxt, shape, mean)
from scipy.fft import rfft, rfftfreq
from datetime import datetime
from decord import VideoReader, cpu, gpu
import frame_extractor as vfe
import detector as d
import plotter as pt
import kivy
kivy.require('2.0.0')


class WindowManager(ScreenManager):
    pass

class Processing(Screen):
    frame_no = 0
    nFrames = 0
    vid_name = ""
    img_path = join(getcwd(), "images", vid_name)
    save_path = join(getcwd(), "results", vid_name)
    frame = ObjectProperty(
        join(img_path, f"{frame_no}.png"))
    frame_processed = ObjectProperty(
        join(img_path, f"processed_{frame_no}.png"))

    fps = 0
    TIMEBAR_YPOS_THRESH = 100
    Hu_dist_thresh = 0.5
    brt_bounds_eye = [0, 0]
    len_bounds_eye = [0, 0]
    brt_bounds_bladder = [0, 0]
    len_bounds_bladder = [0, 0]
    ins_offset_eyeL = [0, 0]
    ins_offset_eyeR = [0, 0]
    ins_offset_bladder = [0, 0]

    settings_names = ["fps",
                      "TIMEBAR_YPOS_THRESH",
                      "Hu_dist_thresh",
                      "brt_bounds_eye",
                      "len_bounds_eye",
                      "brt_bounds_bladder",
                      "len_bounds_bladder",
                      "ins_offset_eyeL",
                      "ins_offset_eyeR",
                      "ins_offset_bladder"]


    def set_properties(self, key):
        try:
            if key=="all":
                keys = ["fps",
                        "ble", "bhe", "cle", "che",
                        "blb", "bhb", "clb", "chb",
                        "ins_off_x_eyeL", "ins_off_y_eyeL",
                        "ins_off_x_eyeR", "ins_off_y_eyeR",
                        "ins_off_x_blad", "ins_off_y_blad",
                        "tpy", "hd"]
            else:
                keys = [key]

            if "fps" in keys:
                self.fps = int(self.ids.fps.text)
            if "ble" in keys:
                self.brt_bounds_eye[0] = int(self.ids.ble.text)
            if "bhe" in keys:
                self.brt_bounds_eye[1] = int(self.ids.bhe.text)
            if "cle" in keys:
                self.len_bounds_eye[0] = int(self.ids.cle.text)
            if "che" in keys:
                self.len_bounds_eye[1] = int(self.ids.che.text)
            if "blb" in keys:
                self.brt_bounds_bladder[0] = int(self.ids.blb.text)
            if "bhb" in keys:
                self.brt_bounds_bladder[1] = int(self.ids.bhb.text)
            if "clb" in keys:
                self.len_bounds_bladder[0] = int(self.ids.clb.text)
            if "chb" in keys:
                self.len_bounds_bladder[1] = int(self.ids.chb.text)
            if "ins_off_x_eyeL" in keys:
                self.ins_offset_eyeL[0] = int(self.ids.ins_off_x_eyeL.text)
            if "ins_off_y_eyeL" in keys:
                self.ins_offset_eyeL[1] = int(self.ids.ins_off_y_eyeL.text)
            if "ins_off_x_eyeR" in keys:
                self.ins_offset_eyeR[0] = int(self.ids.ins_off_x_eyeR.text)
            if "ins_off_y_eyeR" in keys:
                self.ins_offset_eyeR[1] = int(self.ids.ins_off_y_eyeR.text)
            if "ins_off_x_blad" in keys:
                self.ins_offset_bladder[0] = int(self.ids.ins_off_x_blad.text)
            if "ins_off_y_blad" in keys:
                self.ins_offset_bladder[1] = int(self.ids.ins_off_y_blad.text)
            if "tpy" in keys:
                self.TIMEBAR_YPOS_THRESH = int(self.ids.tpy.text)
            if "hd" in keys:
                self.Hu_dist_thresh = float(self.ids.hd.text)
        except:
            pass

    def upload_properties_to_gui(self):
        self.ids.fps.text = str(self.fps)
        self.ids.ble.text = str(self.brt_bounds_eye[0])
        self.ids.bhe.text = str(self.brt_bounds_eye[1])
        self.ids.cle.text = str(self.len_bounds_eye[0])
        self.ids.che.text = str(self.len_bounds_eye[1])
        self.ids.blb.text = str(self.brt_bounds_bladder[0])
        self.ids.bhb.text = str(self.brt_bounds_bladder[1])
        self.ids.clb.text = str(self.len_bounds_bladder[0])
        self.ids.chb.text = str(self.len_bounds_bladder[1])
        self.ids.ins_off_x_eyeL.text = str(self.ins_offset_eyeL[0])
        self.ids.ins_off_y_eyeL.text = str(self.ins_offset_eyeL[1])
        self.ids.ins_off_x_eyeR.text = str(self.ins_offset_eyeR[0])
        self.ids.ins_off_y_eyeR.text = str(self.ins_offset_eyeR[1])
        self.ids.ins_off_x_blad.text = str(self.ins_offset_bladder[0])
        self.ids.ins_off_y_blad.text = str(self.ins_offset_bladder[1])
        self.ids.tpy.text = str(self.TIMEBAR_YPOS_THRESH)
        self.ids.hd.text = str(self.Hu_dist_thresh)

    def frame_extraction(self):
        try:
            vfe.createFolder(self.img_path)
            vfe.createFolder(self.save_path)

            overwrite = False
            # can set to ctx=cpu(0) or ctx=gpu(0)
            vr = VideoReader(
                join("videos", self.vid_name),
                ctx=cpu(0))
            for frame_counter in range(0, self.nFrames):
                file_name = join(
                    self.img_path,
                    f"{frame_counter}.png")
                if (not exists(file_name)) or overwrite:
                    frame = vr[frame_counter]
                    imwrite(file_name,
                            cvtColor(frame.asnumpy(),
                                     COLOR_RGB2BGR))
                else:
                    pass
                print(f"{frame_counter+1} of {self.nFrames} extracted.")
        except:
            print("ERROR: No video is selected")

    def set_vid_name(self):
        vid_path = self.ids.filechooser.selection[0]
        self.vid_name = basename(normpath(vid_path))
        self.frame_no = 0
        self.nFrames = vfe.get_nFrames('videos',
                                       self.vid_name)
        self.ids.progress_frame_extractor.value = 0
        self.ids.progress_frame_extractor.max = self.nFrames - 1
        self.ids.frame_slider.max = self.nFrames - 1
        self.load_images()
        self.ids.vid_selected.text = 'Selected: "' + \
            self.vid_name + '"' + \
            f"  -  total {self.nFrames} frames"
        self.save_path = join(
            getcwd(), "results", self.vid_name)
        self.load_settings()

    def dec_frame(self):
        self.frame_no -= 1
        self.ids.frame_no_input.text = str(self.frame_no)
        self.adjust_slider_pos()
        self.load_images()
    
    def inc_frame(self):
        self.frame_no += 1
        self.ids.frame_no_input.text = str(self.frame_no)
        self.adjust_slider_pos()
        self.load_images()

    def goto_frame(self):
        self.frame_no = int(self.ids.frame_no_input.text)
        self.adjust_slider_pos()
        self.load_images()

    def adjust_slider_pos(self):
        self.ids.frame_slider.value = self.frame_no

    def load_images(self):
        self.img_path = join(getcwd(),
                             "images",
                             self.vid_name)
        self.frame = join(self.img_path,
                          f"{self.frame_no}.png")
        self.frame_processed = join(self.img_path,
                                    f"processed_{self.frame_no}.png")

    def slide(self, *args):
        self.frame_no = int(self.ids.frame_slider.value)
        self.ids.frame_no_input.text = str(self.frame_no)
        self.load_images()

    def save_settings(self):
        vfe.createFolder(self.save_path)
        try:
            dict = {}
            settings = [self.fps,
                        self.TIMEBAR_YPOS_THRESH,
                        self.Hu_dist_thresh,
                        self.brt_bounds_eye,
                        self.len_bounds_eye,
                        self.brt_bounds_bladder,
                        self.len_bounds_bladder,
                        self.ins_offset_eyeL,
                        self.ins_offset_eyeR,
                        self.ins_offset_bladder]
            for item, item_name in zip(settings,
                                       self.settings_names):
                dict[item_name] = item
            file_name = join(self.save_path, "settings.json")
            out_file = open(file_name, "w")
            dump(dict, out_file, indent=4)
            out_file.close()
        except:
            pass

    def load_settings(self):
        try:
            file = join(self.save_path, "settings.json")
            with open(file) as load_settings:
                loaded = load(load_settings)
                self.fps = loaded['fps']
                self.TIMEBAR_YPOS_THRESH = loaded['TIMEBAR_YPOS_THRESH']
                self.Hu_dist_thresh = loaded['Hu_dist_thresh']
                self.brt_bounds_eye = loaded['brt_bounds_eye']
                self.len_bounds_eye = loaded['len_bounds_eye']
                self.brt_bounds_bladder = loaded['brt_bounds_bladder']
                self.len_bounds_bladder = loaded['len_bounds_bladder']
                self.ins_offset_eyeL = loaded['ins_offset_eyeL']
                self.ins_offset_eyeR = loaded['ins_offset_eyeR']
                self.ins_offset_bladder = loaded['ins_offset_bladder']        
            self.upload_properties_to_gui()

        except:
            self.fps = ""
            self.TIMEBAR_YPOS_THRESH = ""
            self.Hu_dist_thresh = ""
            self.brt_bounds_eye = ["", ""]
            self.len_bounds_eye = ["", ""]
            self.brt_bounds_bladder = ["", ""]
            self.len_bounds_bladder = ["", ""]
            self.ins_offset_eyeL = ["", ""]
            self.ins_offset_eyeR = ["", ""]
            self.ins_offset_bladder = ["", ""]
            self.upload_properties_to_gui()

    def detection(self, frame):
        if frame == "all":
            bDebug = False
            file_list = []
            for file_name in listdir(self.img_path):
                if file_name[0].isdigit():
                    file_list.append(file_name)

            if len(file_list) != self.nFrames:
                print("WARNING: Not all frames from the",
                      "selected video are extracted.",
                      "Please extract the frames first.")
            (out_bDetected,
             out_frame_no,
             out_angle_B,
             out_angle_L,
             out_angle_R,
             out_area_L,
             out_area_R,
             out_ax_min_L,
             out_ax_maj_L,
             out_ax_min_R,
             out_ax_maj_R) = d.alloc_result_space(
                 self.nFrames)
            for i, file in enumerate(file_list):
                out_frame_no[i] = d.get_frame_no(file)
                img_input = join(self.img_path,
                                 file)
                img_output = join(self.img_path,
                                  "processed_"+file)
                (out_bDetected[i],
                 out_angle_B[i],
                 out_angle_L[i],
                 out_angle_R[i],
                 out_area_L[i],
                 out_area_R[i],
                 out_ax_min_L[i],
                 out_ax_maj_L[i],
                 out_ax_min_R[i],
                 out_ax_maj_R[i]) = d.main(
                    self.TIMEBAR_YPOS_THRESH,
                    self.brt_bounds_eye,
                    self.len_bounds_eye,
                    self.brt_bounds_bladder,
                    self.len_bounds_bladder,
                    self.Hu_dist_thresh,
                    self.ins_offset_eyeL,
                    self.ins_offset_eyeR,
                    self.ins_offset_bladder,
                    img_input,
                    img_output,
                    bDebug)
                if out_bDetected[i]:
                    print(f"Frame #{int(out_frame_no[i])} of",
                        f"{len(file_list)} succesfully processed.")
                else:
                    print(f"Frame #{int(out_frame_no[i])} of",
                        f"{len(file_list)} failed.")
            self.save_result(
                out_bDetected,
                out_frame_no,
                out_angle_B,
                out_angle_L,
                out_angle_R,
                out_area_L,
                out_area_R,
                out_ax_min_L,
                out_ax_maj_L,
                out_ax_min_R,
                out_ax_maj_R)
            nDetected = count_nonzero(out_bDetected)
            print(f"{self.nFrames - nDetected} out of",
                  f"{self.nFrames} frames failed.")

        # test/debug mode
        else:
            bDebug = True
            (bDetected,
             body_angle,
             eye_angle_L,
             eye_angle_R,
             eye_area_L,
             eye_area_R,
             ax_min_L,
             ax_maj_L,
             ax_min_R,
             ax_maj_R
             ) = d.main(
                 self.TIMEBAR_YPOS_THRESH,
                 self.brt_bounds_eye,
                 self.len_bounds_eye,
                 self.brt_bounds_bladder,
                 self.len_bounds_bladder,
                 self.Hu_dist_thresh,
                 self.ins_offset_eyeL,
                 self.ins_offset_eyeR,
                 self.ins_offset_bladder,
                 self.frame,
                 self.frame_processed,
                 bDebug)
            if(bDetected):
                print(f"body_angle: {body_angle}")
                print(f"eye_angle_L: {eye_angle_L}")
                print(f"eye_angle_R: {eye_angle_R}")
                print(f"eye_area_L: {eye_area_L}")
                print(f"eye_area_R: {eye_area_R}")
                print(f"eye_axis_min_L: {ax_min_L}")
                print(f"eye_axis_maj_L: {ax_maj_L}")
                print(f"eye_axis_min_R: {ax_min_R}")
                print(f"eye_axis_maj_R: {ax_maj_R}")
            self.load_images()

    def normalize_area(self, area):
        mean_area = mean(area)
        return (area - mean_area) / mean_area

    def save_result(
        self, out_bDetected, out_frame_no,
        out_angle_B,out_angle_L, out_angle_R,
        out_area_L, out_area_R,
        out_ax_min_L, out_ax_maj_L,
        out_ax_min_R, out_ax_maj_R):
        out_time = out_frame_no/self.fps
        out_angle_wrtB_L = out_angle_L - out_angle_B
        out_angle_wrtB_R = out_angle_R - out_angle_B
        out_angle_wrtB_L = self.fix_twisted_eyes(out_angle_wrtB_L)
        out_angle_wrtB_R = self.fix_twisted_eyes(out_angle_wrtB_R)
        out_angVel_L = self.get_angVel(out_angle_L, 1/self.fps)
        out_angVel_R = self.get_angVel(out_angle_R, 1/self.fps)
        out_angVel_wrtB_L = self.get_angVel(out_angle_wrtB_L, 1/self.fps)
        out_angVel_wrtB_R = self.get_angVel(out_angle_wrtB_R, 1/self.fps)
        out_area_norm_L = self.normalize_area(out_area_L)
        out_area_norm_R = self.normalize_area(out_area_R)
        out_axes_ratio_L = out_ax_maj_L/out_ax_min_L
        out_axes_ratio_R = out_ax_maj_R/out_ax_min_R

        detection_log = vstack(
            (out_frame_no, out_time, out_bDetected, out_angle_B,
             out_angle_L, out_angle_wrtB_L,
             out_angVel_L, out_angVel_wrtB_L,
             out_angle_R, out_angle_wrtB_R,
             out_angVel_R, out_angVel_wrtB_R,
             out_area_L, out_area_R,
             out_area_norm_L, out_area_norm_R,
             out_ax_min_L, out_ax_maj_L, out_axes_ratio_L,
             out_ax_min_R, out_ax_maj_R, out_axes_ratio_R)).T
        header = "frame_no,time,bDetected,angle_B," + \
            "angle_L,angle_wrtB_L,angVel_L,angVel_wrtB_L," + \
            "angle_R,angle_wrtB_R,angVel_R,angVel_wrtB_R," + \
            "area_L,area_R,area_norm_L,area_norm_R," + \
            "ax_min_L,ax_maj_L,ax_ratio_L," + \
            "ax_min_R,ax_maj_R,ax_ratio_R"
        savetxt(join(self.save_path, "result.csv"),
                detection_log, delimiter=',',
                header=header)
    
    def fix_twisted_eyes(self, angle_list):
        for i, angle in enumerate(angle_list):
            if angle >= 90:
                angle_list[i] = angle-180
        return angle_list

    def get_angVel(self, angle_list, deltaTime):
        angVel_list = angle_list.copy()
        angVel_list[0] = 0
        for i in range(1, len(angVel_list)):
            angVel_list[i] = \
                (angle_list[i] - angle_list[i-1]) / \
                    deltaTime
        return angVel_list

class Plotting(Screen):
    vid_name = ""
    result_path = join(getcwd(), "results", vid_name)
    data_file = ""
    graph_file = ObjectProperty("")
    eye_selection = ["left", "right"]
    axes_selection = {
        "x": "time",
        "y": "angle"}
    custom_label = ["", ""]
    custom_eye_label = ["Left", "Right"]
    custom_grid = [None, None]
    custom_colors = ["blue", "green"]
    graph_title = ""
    wrt_B = True
    x_range = [0, 0]
    y_range = [0, 0]
    c_frame_no = None
    c_time = None
    c_bDetected = None
    c_angle_B = None
    c_angle_L = None
    c_angle_wrtB_L = None
    c_angVel_L = None
    c_angVel_wrtB_L = None
    c_angle_R = None
    c_angle_wrtB_R = None
    c_angVel_R = None
    c_angVel_wrtB_R = None

    def set_custom_colors(self):
        self.custom_colors[0] = self.ids.graph_color_left.text
        self.custom_colors[1] = self.ids.graph_color_right.text
        print("Update - custom colors: ", self.custom_colors)

    def set_graph_title(self):
        self.graph_title = self.ids.graph_title.text
        print("Update - graph title: ", self.graph_title)

    def fetch_x(self):
        if self.axes_selection["x"] == "frame":
            x = self.c_frame_no
            xlabel = "frame"
        elif self.axes_selection["x"] == "time":
            x = self.c_time
            xlabel = "time [sec]"
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
        y_selected = self.axes_selection["y"]
        if "left" in self.eye_selection:
            if y_selected == "area":
                y_list.append(self.c_area_norm_L)
                y_labels.append("Normalized area")
                pass
            if y_selected == "axRatio":
                y_list.append(self.c_ax_ratio_L)
                y_labels.append("Axes ratio (major/minor)")
                pass
            if self.wrt_B == True:
                if y_selected == "angle":
                    y_list.append(self.c_angle_wrtB_L)
                    y_labels.append("angle [degree]")
                elif y_selected == "angVel":
                    y_list.append(self.c_angVel_wrtB_L)
                    y_labels.append("angular velocity [degree/sec]")
                else:
                    pass
            else:
                if y_selected == "angle":
                    y_labels.append("angle [degree]")
                    y_list.append(self.c_angle_L)
                elif y_selected == "angVel":
                    y_list.append(self.c_angVel_L)
                    y_labels.append("angular velocity [degree/sec]")
                else:
                    pass
        if "right" in self.eye_selection:
            if y_selected == "area":
                y_list.append(self.c_area_norm_R)
                y_labels.append("Normalized area")
            if y_selected == "axRatio":
                y_list.append(self.c_ax_ratio_R)
                y_labels.append("Axes ratio (major:minor)")
            if self.wrt_B == True:
                if y_selected == "angle":
                    y_list.append(self.c_angle_wrtB_R)
                    y_labels.append("angle [degree]")
                elif y_selected == "angVel":
                    y_list.append(self.c_angVel_wrtB_R)
                    y_labels.append("angular velocity [degree/sec]")
                else:
                    pass
            else:
                if y_selected == "angle":
                    y_list.append(self.c_angle_R)
                    y_labels.append("angle [degree]")
                elif y_selected == "angVel":
                    y_list.append(self.c_angVel_R)
                    y_labels.append("angular velocity [degree/sec]")
                else:
                    pass
        # two identical labels are appended during 
        # the if/else statements above
        y_label = y_labels[0]
        print(f"{y_label} fetched for y-axis")

        return y_list, y_label

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
                y_part[:-1], bDetected_part[:-1]
            )

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

    def generate_graph(self):
        try:
            x, xlabel = self.fetch_x()
            y, y_label = self.fetch_y()

            for y_arr in y:
                self.interpolate(y_arr)

            if self.axes_selection["x"] == "freq":
                for i, y_arr in enumerate(y):
                    print(shape(y_arr))
                    y[i] = abs(rfft(y_arr / len(y_arr)))
                nSamples = int(max(self.c_frame_no))
                sample_interval = self.c_time[1]
                x = rfftfreq(nSamples, sample_interval)
                y_label += " - Amplitude"
            
            now = datetime.now()
            date_time = now.strftime("%Y_%m_%d-%H_%M_%S.png")
            output = join(
                self.result_path, date_time) 
            pt.main(output,
                    x, xlabel, self.x_range,
                    y, y_label, self.y_range,
                    self.eye_selection,
                    self.custom_grid,
                    self.custom_label,
                    self.custom_eye_label,
                    self.custom_colors,
                    self.graph_title,
                    self.c_bDetected,)
            self.graph_file = output
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
                self.x_range[0] = int(self.ids.x_range_from.text)
            elif idx == "to":
                self.x_range[1] = int(self.ids.x_range_to.text)
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
                self.x_range[0] = int(self.ids.x_range_from.text)
                self.x_range[1] = int(self.ids.x_range_to.text)
        except:
            print("ERROR: Invalid input for x-axis range.")
        print("Update - x_range: ", self.x_range)

    def set_y_range(self, idx):
        try:
            if idx == "from":
                self.y_range[0] = int(self.ids.y_range_from.text)
            elif idx == "to":
                self.y_range[1] = int(self.ids.y_range_to.text)
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
                self.y_range[0] = int(self.ids.y_range_from.text)
                self.y_range[1] = int(self.ids.y_range_to.text)
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
        except:
            pass
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

    def select_vid(self):
        vid_path = self.ids.filechooser.selection[0]
        self.vid_name = basename(normpath(vid_path))
        self.result_path = join(
            getcwd(), "results", self.vid_name)
        self.ids.vid_selected.text = \
            'Selected: "' + self.vid_name + '"'
        self.data_file = join(self.result_path, "result.csv")

        default_graph = join(
            self.result_path, ".blank.png")
        if not exists(default_graph):
            pt.generate_blank(default_graph)
        self.graph_file = default_graph
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
            self.c_angle_B = data_frame[:, 3]
            self.c_angle_L = data_frame[:, 4]
            self.c_angle_wrtB_L = data_frame[:, 5]
            self.c_angVel_L = data_frame[:, 6]
            self.c_angVel_wrtB_L = data_frame[:, 7]
            self.c_angle_R = data_frame[:, 8]
            self.c_angle_wrtB_R = data_frame[:, 9]
            self.c_angVel_R = data_frame[:, 10]
            self.c_angVel_wrtB_R = data_frame[:, 11]
            self.c_area_L = data_frame[:, 12]
            self.c_area_R = data_frame[:, 13]
            self.c_area_norm_L = data_frame[:, 14]
            self.c_area_norm_R = data_frame[:, 15]
            self.c_ax_min_L = data_frame[:, 16]
            self.c_ax_maj_L = data_frame[:, 17]
            self.c_ax_ratio_L = data_frame[:, 18]
            self.c_ax_min_R = data_frame[:, 19]
            self.c_ax_maj_R = data_frame[:, 20]
            self.c_ax_ratio_R = data_frame[:, 21]

        except:
            print("ERROR: Processing result data not available.")

Builder.load_file('zebrafish.kv')
Builder.load_file('processing.kv')
Builder.load_file('plotting.kv')
wm_kv = Builder.load_file('window_manager.kv')

class ZebrafishApp(App):
    def build(self):
        return wm_kv

if __name__ == '__main__':
    Window.size = (1200, 900)
    Window.top = 50
    Window.left = 100
    ZebrafishApp().run()
