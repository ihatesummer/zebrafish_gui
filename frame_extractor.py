# %%
from cv2 import VideoCapture, imwrite, waitKey, cvtColor
from cv2 import COLOR_RGB2BGR, CAP_PROP_FRAME_COUNT
from os import makedirs, listdir, getcwd
from os.path import exists, join
from decord import VideoReader, cpu, gpu


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


def main_decord(vid: str, IMG_PATH: str,
                SAVE_PATH:str ,VID_PATH: str,
                overwrite=False) -> None:
    """
    Extract frames from a video using decord's VideoReader. Faster than openCV
    :param vid: file name of the video
    :param IMAGE_PATH: the directory to save the frames
    :param VIDEO_PATH: the directory where the video file exists
    :param overwrite: to overwrite frames that already exist?
    """

    createFolder(IMG_PATH)
    createFolder(SAVE_PATH)

    # can set to ctx=cpu(0) or ctx=gpu(0)
    vr = VideoReader(join(VID_PATH, vid), ctx=cpu(0))
    start = 0
    nFrames = get_nFrames(VID_PATH, vid)
    for frame_counter in range(start, nFrames):
        file_name = join(IMG_PATH, f"{frame_counter}.png")
        if (not exists(file_name)) or overwrite:
            frame = vr[frame_counter]
            imwrite(file_name, cvtColor(frame.asnumpy(), COLOR_RGB2BGR))
            print(f"Frame {frame_counter} of {nFrames-1} saved")
        else:
            print(f"Frame {frame_counter} of {nFrames-1} skipped")


if __name__ == "__main__":
    print("WARNING: this is not the main module.")
