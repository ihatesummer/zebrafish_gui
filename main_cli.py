# %%
import video_frame_extraction as frame_extractor
import angle_detection as angle_detector
from os import getcwd, listdir
from os.path import join
from json import dump


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def save_settings():
    dict = {}
    settings = [TIMEBAR_YPOS_THRESH,
                brt_bounds_eye,
                len_bounds_eye,
                brt_bounds_bladder,
                len_bounds_bladder,
                inscription_pos_offset_eyeL,
                inscription_pos_offset_eyeR,
                inscription_pos_offset_bladder]
    for item in settings:
        attribute_name = namestr(item, globals())[0]
        dict[attribute_name] = item
    file_name = join(DEST_PATH, "settings.json")
    out_file = open(file_name, "w")
    dump(dict, out_file, indent=4)
    out_file.close()


TIMEBAR_YPOS_THRESH = 100
brt_bounds_eye = [0, 50]
len_bounds_eye = [150, 300]
brt_bounds_bladder = [0, 90]
len_bounds_bladder = [150, 500]
# remove false eye(s) if Hu distance is greater than this threshold
Hu_dist_thresh = 0.5
# B13WT types
inscription_pos_offset_eyeL = (-100, -30)
inscription_pos_offset_eyeR = (20, -30)
inscription_pos_offset_bladder = (30, 0)
CWD = getcwd()
vid = input("video file name?"
            "\n- you can input 'all' to"
            "iterate through all videos."
            "\n- you can input 'debug' to"
            "use debug mode."
            "\n:")

VID_PATH = join(CWD, 'video')
IMG_PATH = join(CWD, 'image')

if vid == "all":
    vids = listdir(VID_PATH)
    bDebug = False
elif vid == 'debug':
    vids = [vid]
    bDebug = True
else:
    vids = [vid]
    bDebug = False


for vid in vids:
    # frame_extractor.main_decord(vid, IMG_PATH, VID_PATH)
    angle_detector.main(vid, IMG_PATH,
                        TIMEBAR_YPOS_THRESH,
                        brt_bounds_eye, len_bounds_eye,
                        brt_bounds_bladder, len_bounds_bladder,
                        Hu_dist_thresh,
                        inscription_pos_offset_eyeL,
                        inscription_pos_offset_eyeR,
                        inscription_pos_offset_bladder,
                        bDebug)
    save_settings()
