import detector as d
import numpy as np
from kivy.uix.screenmanager import Screen
from decord import VideoReader, cpu, gpu
from json import dump, load
from re import findall
from os import getcwd, listdir, makedirs
from os.path import exists, join, basename, normpath
from cv2 import *

class Processor_Window(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_no = 0
        self.nFrames = 0
        self.vid_name = ""
        self.img_path = join(getcwd(), "images", self.vid_name)
        self.save_path = join(getcwd(), "results", self.vid_name)
        self.frame = join(
            self.img_path, f"{self.frame_no}.png")
        self.frame_processed = join(
            self.img_path, f"processed_{self.frame_no}.png")
        print(self.img_path, type(self.img_path))
        print(self.frame_processed, type(self.frame_processed))

        self.fps = 0
        self.TIMEBAR_YPOS_THRESH = 100
        self.Hu_dist_thresh = 0.5
        self.brt_bounds_eye = [0, 0]
        self.len_bounds_eye = [0, 0]
        self.brt_bounds_bladder = [0, 0]
        self.len_bounds_bladder = [0, 0]
        self.ins_offset_eyeL = [0, 0]
        self.ins_offset_eyeR = [0, 0]
        self.ins_offset_bladder = [0, 0]
        self.crop_ratio = [[0, 1], [0, 1]]
        self.bBladderSkip = False
        self.settings_names = ["fps",
                               "crop_width",
                               "crop_height",
                               "Hu_dist_thresh",
                               "brt_bounds_eye",
                               "len_bounds_eye",
                               "brt_bounds_bladder",
                               "len_bounds_bladder",
                               "ins_offset_eyeL",
                               "ins_offset_eyeR",
                               "ins_offset_bladder"]

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
    
    def refresh_filechooser(self):
        self.ids.filechooser._update_files()
        print("Filechooser refreshed.")

    def update_bladder_skip(self, instance, value):
        self.bBladderSkip = value
        print("Update - Bladder Skip: ", self.bBladderSkip)

    def set_properties(self, key):
        try:
            if key == "all":
                keys = ["fps",
                        "ble", "bhe", "cle", "che",
                        "blb", "bhb", "clb", "chb",
                        "ins_off_x_eyeL", "ins_off_y_eyeL",
                        "ins_off_x_eyeR", "ins_off_y_eyeR",
                        "ins_off_x_blad", "ins_off_y_blad",
                        "hd",
                        "crp_x_l",
                        "crp_x_h",
                        "crp_y_l",
                        "crp_y_h"]
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
            if "hd" in keys:
                self.Hu_dist_thresh = float(self.ids.hd.text)
            if "crp_x_l" in keys:
                self.crop_ratio[0][0] = float(self.ids.crp_x_l.text)
            if "crp_x_h" in keys:
                self.crop_ratio[0][1] = float(self.ids.crp_x_h.text)
            if "crp_y_l" in keys:
                self.crop_ratio[1][0] = float(self.ids.crp_y_l.text)
            if "crp_y_h" in keys:
                self.crop_ratio[1][1] = float(self.ids.crp_y_h.text)
        except Exception as e:
            print(f"ERROR: {e}")

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
        self.ids.hd.text = str(self.Hu_dist_thresh)
        self.ids.crp_x_l.text = str(self.crop_ratio[0][0])
        self.ids.crp_x_h.text = str(self.crop_ratio[0][1])
        self.ids.crp_y_l.text = str(self.crop_ratio[1][0])
        self.ids.crp_y_h.text = str(self.crop_ratio[1][1])

    def frame_extraction(self):
        try:
            self.createFolder(self.img_path)
            self.createFolder(self.save_path)

            overwrite = False
            # can set to ctx=cpu(0) or ctx=gpu(0)
            vr = VideoReader(
                join("videos", self.vid_name), ctx=cpu(0))
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
        
        self.ids.preview_orig.reload()

    def get_nFrames(self, VID_PATH: str, vid: str) -> int:
        """
        Get the total number of frames of a given video file
        :param VID_PATH: folder containing the video file
        :param vid: name of the video file
        """
        vc = VideoCapture(join(VID_PATH, vid))
        nFrames = int(vc.get(CAP_PROP_FRAME_COUNT)) - 1
        return nFrames

    def set_vid_name(self):
        vid_path = self.ids.filechooser.selection[0]
        self.vid_name = basename(normpath(vid_path))
        self.frame_no = 0
        self.nFrames = self.get_nFrames('videos',
                                        self.vid_name)
        self.ids.frame_slider.max = self.nFrames - 1
        self.load_images()
        self.ids.vid_selected.text = \
            f"[color=aaff00]{self.vid_name}[/color] ({self.nFrames} frames) selected."
        self.save_path = join(
            getcwd(), "results", self.vid_name)
        self.load_settings()
        print(f"Video {self.vid_name} selected.")

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
        self.ids.preview_orig.source = self.frame
        self.ids.preview_proc.source = self.frame_processed

    def slide(self, *args):
        self.frame_no = int(self.ids.frame_slider.value)
        self.ids.frame_no_input.text = str(self.frame_no)
        self.load_images()

    def save_settings(self):
        self.createFolder(self.save_path)
        try:
            dict = {}
            settings = [self.fps,
                        self.crop_ratio[0],
                        self.crop_ratio[1],
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
        except Exception as e:
            print(f"ERROR saving the settings: {e}")

    def load_settings(self):
        specific_setting = join(self.save_path, "settings.json")
        if exists(specific_setting):
            file = specific_setting
            print(f"Specific settings loaded.")
        else:
            file = "default_settings.json"
            print(f"Default settings loaded.")
        with open(file) as load_settings:
            loaded = load(load_settings)
            self.fps = loaded['fps']
            self.Hu_dist_thresh = loaded['Hu_dist_thresh']
            self.crop_ratio[0] = loaded['crop_width']
            self.crop_ratio[1] = loaded['crop_height']
            self.brt_bounds_eye = loaded['brt_bounds_eye']
            self.len_bounds_eye = loaded['len_bounds_eye']
            self.brt_bounds_bladder = loaded['brt_bounds_bladder']
            self.len_bounds_bladder = loaded['len_bounds_bladder']
            self.ins_offset_eyeL = loaded['ins_offset_eyeL']
            self.ins_offset_eyeR = loaded['ins_offset_eyeR']
            self.ins_offset_bladder = loaded['ins_offset_bladder']        
        self.upload_properties_to_gui()

    def alloc_result_space(self, nFrames):
        # bool, True by default
        out_bDetected = np.zeros(nFrames) < 1
        # int, from 0 to nFrames-1
        out_frame_no = np.zeros(nFrames)
        # float, body eye angle [degree]
        out_angle_B = np.zeros(nFrames)
        # float, left eye angle [degree]
        out_angle_L = np.zeros(nFrames)
        # float, right eye angle [degree]
        out_angle_R = np.zeros(nFrames)
        # float, left eye area
        out_area_L = np.zeros(nFrames)
        # float, right eye area
        out_area_R = np.zeros(nFrames)
        # float, left eye minor axis length
        out_ax_min_L = np.zeros(nFrames)
        # float, left eye major axis length
        out_ax_maj_L = np.zeros(nFrames)
        # float, right eye minor axis length
        out_ax_min_R = np.zeros(nFrames)
        # float, right eye major axis length
        out_ax_maj_R = np.zeros(nFrames)

        return (out_bDetected, out_frame_no,
                out_angle_B,
                out_angle_L, out_angle_R,
                out_area_L, out_area_R,
                out_ax_min_L, out_ax_maj_L,
                out_ax_min_R, out_ax_maj_R)

    def get_frame_no(self, filename: str) -> int:
        """
        Extracts number from a given filename.
        Error when the filename contains more than one numbers.
        """
        num = findall(r'\d+', filename)
        if len(num) == 1:
            return int(num[0])
        else:
            print("ERROR: Can't retrieve frame number ; \
                filename contains more than one number")
            return None

    def detection(self, frame, debug):
        if frame == "all":
            file_list = []
            try:
                for file_name in listdir(self.img_path):
                    bProcessedfile = not file_name[0].isdigit()
                    # file_name[:-4] through file_name[:-1] = ".png"
                    bDebugFile = not file_name[-5].isdigit()
                    if not (bProcessedfile or bDebugFile):
                        file_list.append(file_name)
            except:
                return None
            if len(file_list) != self.nFrames:
                print("WARNING: Not all frames from the",
                      "selected video are extracted.",
                      "Please extract the frames first.")
            (out_bDetected,
             out_frame_no,
             out_angle_B,
             out_angle_L,
             out_angle_R,
             area_L,
             area_R,
             ax_min_L,
             ax_maj_L,
             ax_min_R,
             ax_maj_R) = self.alloc_result_space(
                self.nFrames)
            for i, file in enumerate(file_list):
                out_frame_no[i] = self.get_frame_no(file)
                img_input = join(self.img_path,
                                 file)
                img_output = join(self.img_path,
                                  "processed_"+file)
                try:
                    (out_bDetected[i],
                    out_angle_B[i],
                    out_angle_L[i],
                    out_angle_R[i],
                    area_L[i],
                    area_R[i],
                    ax_min_L[i],
                    ax_maj_L[i],
                    ax_min_R[i],
                    ax_maj_R[i]) = d.main(
                        self.crop_ratio,
                        self.bBladderSkip,
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
                        debug)
                except:
                    (out_bDetected[i],
                     out_angle_B[i],
                     out_angle_L[i],
                     out_angle_R[i],
                     area_L[i],
                     area_R[i],
                     ax_min_L[i],
                     ax_maj_L[i],
                     ax_min_R[i],
                     ax_maj_R[i]) = (False, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0)
                
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
                area_L,
                area_R,
                ax_min_L,
                ax_maj_L,
                ax_min_R,
                ax_maj_R)
            nDetected = np.count_nonzero(out_bDetected)
            print(f"{self.nFrames - nDetected} out of",
                  f"{self.nFrames} frames failed.")
        # test/debug mode
        else:
            try:
                savename = d.main(
                    self.crop_ratio,
                    self.bBladderSkip,
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
                    debug
                    )
                self.ids.preview_proc.source = savename
                self.ids.preview_proc.reload()
            except Exception as e:
                print(f"ERROR: try adjusting the settings: {e}")

    def normalize_area(self, area):
        mean_area = np.mean(area)
        return (area - mean_area) / mean_area

    def save_result(
        self, out_bDetected, out_frame_no,
        out_angle_B, out_angle_L, out_angle_R,
        area_L, area_R,
        ax_min_L, ax_maj_L,
        ax_min_R, ax_maj_R):

        out_time = out_frame_no/self.fps
        out_angle_wrtB_L = out_angle_L - out_angle_B
        out_angle_wrtB_R = out_angle_R - out_angle_B
        out_angle_wrtB_L = self.fix_twisted_eyes(out_angle_wrtB_L)
        out_angle_wrtB_R = self.fix_twisted_eyes(out_angle_wrtB_R)
        # invert eye angle w.r.t. body
        # so that pos. value is to the right
        # and neg. value is to the left
        out_angle_wrtB_L *= -1
        out_angle_wrtB_R *= -1
        out_area_norm_L = self.normalize_area(area_L)
        out_area_norm_R = self.normalize_area(area_R)
        out_axes_ratio_L = ax_maj_L/ax_min_L
        out_axes_ratio_R = ax_maj_R/ax_min_R

        detection_log = np.vstack(
            (out_frame_no, out_time,
             out_bDetected, out_angle_B,
             out_angle_L, out_angle_wrtB_L,
             out_angle_R, out_angle_wrtB_R,
             out_area_norm_L, out_area_norm_R,
             out_axes_ratio_L, out_axes_ratio_R)).T
        header = "frame_no,time,bDetected,angle_B," + \
            "angle_L,angle_wrtB_L," + \
            "angle_R,angle_wrtB_R," + \
            "area_norm_L,area_norm_R," + \
            "ax_ratio_L,ax_ratio_R"
        np.savetxt(join(self.save_path, "result.csv"),
                detection_log, delimiter=',',
                header=header)
    
    def fix_twisted_eyes(self, angle_list):
        for i, angle in enumerate(angle_list):
            if angle >= 90:
                angle_list[i] = angle-180
        return angle_list
