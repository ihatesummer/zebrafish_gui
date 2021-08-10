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


def main(vid: str, IMAGE_PATH: str, VIDEO_PATH: str) -> None:
    """
    Extract frames from a video using openCV.
    :param vid: file name of the video
    :param IMAGE_PATH: the directory to save the frames
    :param VIDEO_PATH: the directory where the video file exists
    """

    img_dest_dir = join(IMAGE_PATH, vid)
    createFolder(img_dest_dir)
    createFolder(img_dest_dir+"_processed")
    save_dest_dir = join(getcwd(), vid)
    createFolder(save_dest_dir)

    # read first frame
    frame_counter = 0
    vidcap = VideoCapture(join(VIDEO_PATH, vid))
    success, image = vidcap.read()
    # write output and extract more frames
    while success:
        file_name = join(img_dest_dir, f'frame{frame_counter}.png')
        imwrite(file_name, image)
        success, image = vidcap.read()
        frame_counter += 1

        # break if 1 second delay (end of the video)
        if waitKey(1000) != -1:
            break


def get_nFrames(VID_PATH, vid):
    vc = VideoCapture(join(VID_PATH, vid))
    nFrames = int(vc.get(CAP_PROP_FRAME_COUNT)) - 1
    return nFrames

if __name__ == "__main__":
    print("WARNING: this is not the main module.")
