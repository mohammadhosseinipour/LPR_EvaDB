import evadb
import shutil
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import requests
warnings.filterwarnings("ignore")

HOME = os.getcwd()
input_type=None
print("""Note that the input should be in size 640 for best performance!\n tip: Use Roboflow!""")
FUNC_FOLDER = str(
            input("""ATTENTION! first add the yolo_object_detection.py file to evadb functions folder before continuing!
    Also change the location to best_LP.pt file in line 27 in yolo_object_detection.py to run correctly.
    Then please enter the path to functions folder(only the folder path) here:""")
        )
if FUNC_FOLDER=="":
    FUNC_FOLDER = "/Users/mohammadhp/anaconda3/envs/evadb-venv/lib/python3.11/site-packages/evadb/functions"
DEFAULT_VIDEO_LINK = "https://www.dropbox.com/scl/fi/41zosrmwg1asbjfazsts8/test2.mp4?rlkey=r3dlhhmxs63b4x7drv02x4fj3&dl=1"
DEFAULT_VIDEO_PATH = HOME+"/LPR_Video/test2.mp4"
DEFAULT_IMAGE_LINK = "https://www.dropbox.com/scl/fi/hjvxasabltqk3f2fblou2/000001.jpg?rlkey=4l3x4e1w5nmtbjfmcp74g5v2q&dl=1"
DEFAULT_IMAGE_PATH = HOME+"/LPR_Image/000001.jpg"
video_folder_path = HOME+"/LPR_Video_temp"
image_folder_path = HOME+"/LPR_Image_temp"



def receive_user_input(cursor):

    global input_type
    print(
        "Welcome! This app lets you to Recognise the License Plates from either several Videos or Images \n\n"
    )
    done=True

    input_type = str(
            input(
                "you want to add Video or Image?(for image wirte -> Img, for video write -> Vid) "
            )
        ).lower()

    online_doc=[]
    offline_doc=[]
    print("input_type:"+str(input_type))
    while done:
        input_loc = str(
            input(
                """you want to add the input online or locally?
                (for online write -> Online,for locally write -> Local, no input? -> Done)"""
            )
        ).lower()

        if input_type=="img":

            if input_loc=="online":
                image_link = str(
                input(
                    """make sure the link if for the image itself not its preview.
                    📺 Enter the URL of the image (press Enter to use our default image URL): """
                    )
                )
                if image_link == "":
                    image_link = DEFAULT_IMAGE_LINK
                online_doc.append(image_link)

            elif input_loc=="local":
                image_path = str(
                input(
                    "📺 Enter the path of the image (press Enter to use our default image path): "
                    )
                )
                if image_path == "":
                    image_path = DEFAULT_IMAGE_PATH
                offline_doc.append(image_path)
            else:
                break
        elif input_type=="vid":
            if input_loc=="online":
                video_link = str(
                input(
                    """make sure the link if for the video itself not its preview.
                    📺 Enter the URL of the video (press Enter to use our default video URL): """
                    )
                )
                if video_link == "":
                    video_link = DEFAULT_VIDEO_LINK
                online_doc.append(video_link)

            elif input_loc=="local":
                video_path = str(
                input(
                    "📺 Enter the path of the video (press Enter to use our default video path): "
                    )
                )
                if video_path == "":
                    video_path = DEFAULT_VIDEO_PATH
                offline_doc.append(video_path)
            else:
                break
    if input_type=="img":
        cursor.query("DROP TABLE IF EXISTS ImageTable;").df()
        cursor=Add_Image_Locally(cursor, offline_doc)
        cursor=Add_Image_Online(cursor, online_doc)
    elif input_type=="vid":
        cursor.query("DROP TABLE IF EXISTS VideoTable;").df()
        cursor=Add_Video_Locally(cursor, offline_doc)
        cursor=Add_Video_Online(cursor, online_doc)
    return cursor
def Add_Video_Locally(cursor, locs):
    for i in range(len(locs)):
        cursor.query(f"LOAD VIDEO '{locs[i]}' INTO VideoTable").df()
    print("video offline load done!")
    return cursor

def Add_Video_Online(cursor, locs):
    create_folder(video_folder_path)
    for name ,url in enumerate(locs):
        download_video_from_url(url, video_folder_path, str(name)+'.mp4')
    files_and_directories = os.listdir(video_folder_path)
    # List of common video extensions. You can add or remove extensions as needed.
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv']
    only_video_files = [f for f in files_and_directories if os.path.isfile(os.path.join(video_folder_path, f)) and any(f.endswith(ext) for ext in video_extensions)]
    for i in range(len(only_video_files)):
        file_path = os.path.join(video_folder_path, only_video_files[i])
        cursor.query(f"LOAD VIDEO '{file_path}' INTO VideoTable").df()
    print("video online load done!")
    return cursor

def Add_Image_Locally(cursor, locs):
    for i in range(len(locs)):
        cursor.query(f"LOAD IMAGE '{locs[i]}' INTO ImageTable").df()
    print("image offline load done!")
    return cursor

def Add_Image_Online(cursor, locs):
    create_folder(image_folder_path)
    for name ,url in enumerate(locs):
        download_image_from_url(url, image_folder_path, str(name)+'.png')
    files_and_directories = os.listdir(image_folder_path)
    # List of common image extensions. You can add or remove extensions as needed.
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    only_image_files = [f for f in files_and_directories if os.path.isfile(os.path.join(image_folder_path, f)) and any(f.lower().endswith(ext) for ext in image_extensions)]
    for i in range(len(only_image_files)):
        file_path = os.path.join(image_folder_path, only_image_files[i])
        cursor.query(f"LOAD IMAGE '{file_path}' INTO ImageTable").df()
    print("image online load done!")
    return cursor

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def download_video_from_url(url, save_folder, video_name=None):
    if video_name is None:
        video_name = url.split('/')[-1]
    save_path = os.path.join(save_folder, video_name)
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
def download_image_from_url(url, save_folder, image_name=None):
    if image_name is None:
        image_name = url.split('/')[-1]
    save_path = os.path.join(save_folder, image_name)
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def Creating_LPR_YOLO_Function(cursor):
    cursor.query("DROP FUNCTION IF EXISTS YoloObjectDetection;").df()
    Yolo_Python_Code="/yolo_object_detection.py"
    cursor.query(f"CREATE FUNCTION YoloObjectDetection IMPL  '{FUNC_FOLDER+Yolo_Python_Code}';").df()
    return cursor

def YOLO_Predict_Function(cursor):
    global input_type
    try:
        Frames = int(
                input("""How many frames/images you want to process?
                (press enter for whole input data)""")
            )
        if input_type=="img":
            result = cursor.query(f"SELECT YoloObjectDetection(data) FROM ImageTable WHERE id<{Frames} ;").df()
        elif input_type=="vid":
            result = cursor.query(f"SELECT YoloObjectDetection(data) FROM VideoTable WHERE id<{Frames} ;").df()
    except:
        if input_type=="img":
            result = cursor.query(f"SELECT YoloObjectDetection(data) FROM ImageTable ;").df()
        elif input_type=="vid":
            result = cursor.query(f"SELECT YoloObjectDetection(data) FROM VideoTable ;").df()
    result=result[result['yoloobjectdetection.scores'].astype(bool)]
    result.reset_index(inplace=True, drop=True)
    return result, cursor



def Display_Plates(result):
    result=result[result['yoloobjectdetection.scores'].astype(bool)]
    result.reset_index(inplace=True, drop=True)
    # you can add img_hight to change the plates size
    # also by changing alpha from 0 to 1 to adjust sharpening
    Display_Cropped_Images(result)

def Display_Cropped_Images(df: pd.DataFrame, img_height=2.5, alpha = 0.7):
    """
    Display the cropped images from the DataFrame.

    Arguments:
    - df (pd.DataFrame): DataFrame containing columns "labels", "scores", and "cropped_images" with image data.
    """

    num_images = len(df)
    if num_images==0:
        print("NO PLATE FOUND!")
        return
    total_height = img_height * num_images
    fig, axs = plt.subplots(nrows=num_images, ncols=1, figsize=(2.5, total_height))

    # If there's only one image, axs is not a list
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, (_, row) in enumerate(df.iterrows()):
        cropped_img = row['yoloobjectdetection.cropped_images']
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(cropped_img, -1, kernel)
        blended = cv2.addWeighted(cropped_img, alpha, sharpened, 1 - alpha, 0)
        denoised_image = cv2.medianBlur(blended, ksize=3)
#         enlarged_img = cv2.resize(cropped_img, (300, 75), interpolation=cv2.INTER_LINEAR)
        axs[i].imshow(denoised_image.astype(np.uint8))
        axs[i].axis('off')  # hide axis

    plt.tight_layout()
    plt.show()




def Del_Func(cursor):
    cursor.query("DROP TABLE IF EXISTS ImageTable;").df()
    cursor.query("DROP TABLE IF EXISTS VideoTable;").df()
    cursor.query("DROP FUNCTION IF EXISTS YoloObjectDetection;").df()

def Clear_Func(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")



def main():
    cursor = evadb.connect().cursor()

    cursor = receive_user_input(cursor)

    cursor = Creating_LPR_YOLO_Function(cursor)

    result, cursor = YOLO_Predict_Function(cursor)

    Display_Plates(result)

    Del_Func(cursor)

    Clear_Func(video_folder_path)
    Clear_Func(image_folder_path)


if __name__ == "__main__":
    main()
