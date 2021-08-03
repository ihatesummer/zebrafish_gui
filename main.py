from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.lang import Builder
from os import getcwd, listdir
from os.path import join, normpath, basename
from json import dump, load
from numpy import vstack, savetxt
import frame_extractor as vfe
import angle_detector as ad
import kivy
kivy.require('2.0.0')

Builder.load_file('zebrafish.kv')
Builder.load_file('processing.kv')

class MainWindow(BoxLayout):
    frame = ObjectProperty("0.png")
    frame_processed = ObjectProperty("processed_0.png")
    text_color = ObjectProperty([0, 0, 0, 1])
    text_size_header = 20
    text_size_label = 10

    main = BoxLayout(orientation='vertical')
    row1 = BoxLayout(orientation='horizontal')
    row2 = BoxLayout(orientation='horizontal')
    main.add_widget(row1)
    main.add_widget(row2)
    # row1.add_widget(ProcessingWindow)
    # proc_window = ProcessingWindow()


class Processing(Widget):
    frame_no = 0
    nFrames = 0
    vid_name = ""
    img_path = join(getcwd(), "images", vid_name)
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
                self.Hu_dist_thresh = int(self.ids.hd.text)
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
            vfe.main_decord(self.vid_name,
                            self.img_path,
                            'videos',
                            overwrite=True)
            self.nFrames = vfe.get_nFrames('videos',
                                        self.vid_name)
        except:
            print("ERROR: No video is selected")

    def set_vid_name(self):
        vid_path = self.ids.filechooser.selection[0]
        self.vid_name = basename(normpath(vid_path))
        self.frame_no = 0
        self.nFrames = vfe.get_nFrames('videos',
                                       self.vid_name)
        self.ids.frame_slider.max = self.nFrames - 1
        self.load_images()
        self.ids.vid_selected.text = 'Selected: "' + \
            self.vid_name + '"' + \
            f"  -  total {self.nFrames} frames"
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
            file_name = join(self.img_path, "settings.json")
            out_file = open(file_name, "w")
            dump(dict, out_file, indent=4)
            out_file.close()
        except:
            pass

    def load_settings(self):
        try:
            file = join(self.img_path, "settings.json")
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
            pass

    def angle_detection(self, frame):
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
             out_angle_L,
             out_angle_R,
             out_angle_B) = ad.alloc_result_space(
                 self.nFrames)
            for i, file in enumerate(file_list):
                out_frame_no[i] = ad.get_frame_no(file)
                img_input = join(self.img_path,
                                 file)
                img_output = join(self.img_path,
                                  "processed_"+file)
                (out_bDetected[i],
                 out_angle_B[i],
                 out_angle_L[i],
                 out_angle_R[i]) = ad.main(
                    self.img_path,
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
                out_angle_L,
                out_angle_R,
                out_angle_B)

        # test/debug mode
        else:
            bDebug = True
            (bDetected,
             body_angle,
             eye_angle_L,
             eye_angle_R
             ) = ad.main(
                 self.img_path,
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
            print(bDetected, body_angle, eye_angle_L, eye_angle_R)

    def save_result(self, out_bDetected, out_frame_no,
                    out_angle_L, out_angle_R, out_angle_B):
        out_time = out_frame_no/self.fps
        out_angle_wrtB_L = out_angle_L - out_angle_B
        out_angle_wrtB_R = out_angle_R - out_angle_B
        out_angle_wrtB_L = self.fix_twisted_eyes(out_angle_wrtB_L)
        out_angle_wrtB_R = self.fix_twisted_eyes(out_angle_wrtB_R)
        out_angVel_L = self.get_angVel(out_angle_L)
        out_angVel_R = self.get_angVel(out_angle_R)
        out_angVel_wrtB_L = self.get_angVel(out_angle_wrtB_L)
        out_angVel_wrtB_R = self.get_angVel(out_angle_wrtB_R)
        detection_log = vstack(
            (out_frame_no, out_time, out_bDetected, out_angle_B,
             out_angle_L, out_angle_wrtB_L,
             out_angVel_L, out_angVel_wrtB_L,
             out_angle_R, out_angle_wrtB_R,
             out_angVel_R, out_angVel_wrtB_R)).T
        header = "frame_no, time, bDetected, angle_B, \
            angle_L, angle_wrtB_L, out_angVel_L, angVel_wrtB_L, \
            angle_R, angle_wrtB_R, out_angVel_R, angVel_wrtB_R"
        savetxt(join(self.img_path, "result.csv"),
                detection_log, delimiter=',',
                header=header)
    
    def fix_twisted_eyes(self, angle_list):
        for i, angle in enumerate(angle_list):
            if angle >= 90:
                angle_list[i] = angle-180
        return angle_list

    def get_angVel(self, angle_list):
        angVel_list = angle_list.copy()
        angVel_list[0] = 0
        for i in range(1, len(angVel_list)):
            angVel_list[i] = angle_list[i] - angle_list[i-1]
        return angVel_list

class Data(BoxLayout):
    pass

class Graph(Widget):
    pass


class ZebrafishApp(App):
    def build(self):
        return Processing()


if __name__ == '__main__':
    Window.size = (1500, 900)
    Window.top = 100
    Window.left = 100
    ZebrafishApp().run()
