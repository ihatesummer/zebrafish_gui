# %%
from cv2 import VideoCapture, imwrite, waitKey, CAP_PROP_FRAME_COUNT
from os import makedirs, getcwd
from os.path import exists, join


def createFolder(dir: str) -> None:
    """
    Create a directory if it does not already exists.
    :param dir: the desired directory name
    """
    try:
        if not exists(dir):
            makedirs(dir)
    except OSError:
        print('Error: Creating directory ' + dir + ' failed. [OSError]')


def get_nFrames(VID_PATH: str, vid: str) -> int:
    """
    Get the total number of frames of a given video file
    :param VID_PATH: folder containing the video file
    :param vid: name of the video file
    """
    vc = VideoCapture(join(VID_PATH, vid))
    nFrames = int(vc.get(CAP_PROP_FRAME_COUNT)) - 1
    return nFrames


if __name__ == "__main__":
    print("WARNING: this is not the main module.")
