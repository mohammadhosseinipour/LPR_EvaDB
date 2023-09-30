import evadb
import shutil
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import requests
import tensorflow as tf
from tensorflow import keras
warnings.filterwarnings("ignore")

print("""Note that the input should be in size 640 for best performance!\n tip: Use Roboflow!""")

HOME = os.getcwd()
CARS_PLATES=[]
CARS_PLATES_PIC=[]
sim_plates=[]
nine_plates=[]
plate_pic=[]
input_type=None
model=None
class_names=["0","1","2","3","4","5","6","7","8","9","A","B","D",
          "Gh","H","J","L","M","N","P","PuV","PwD","Sad","Sin",
          "T","Taxi","V","Y"]

plate_checklist=[[True,1],[True,1],[True,2],[True,1],[True,1],[True,1],[True,1],[True,1]]
model_path="character_recognition_model"

FUNC_FOLDER = str(
            input("""ATTENTION! first add the yolo_object_detection.py file to evadb functions folder before continuing!
    Also change the location to best_LP.pt file in line 27 in yolo_object_detection.py to run correctly.
    Then please enter the path to functions folder(only the folder path) here:""")
        )
if FUNC_FOLDER=="":
    FUNC_FOLDER = "/Users/mohammadhp/anaconda3/envs/evadb-venv/lib/python3.11/site-packages/evadb/functions"
DEFAULT_VIDEO_LINK = "https://www.dropbox.com/scl/fi/41zosrmwg1asbjfazsts8/test2.mp4?rlkey=r3dlhhmxs63b4x7drv02x4fj3&dl=1"
DEFAULT_VIDEO_PATH = HOME+"/LPR_Video/test2.mp4"
DEFAULT_IMAGE_LINK = "https://www.dropbox.com/scl/fi/1nls2y7neow6x42yra6ak/000005.png?rlkey=gwabhlhr3m6tgl1svgwcxt6xv&dl=1"
DEFAULT_IMAGE_PATH = HOME+"/LPR_Image/000005.png"
video_folder_path = HOME+"/LPR_Video_temp"
image_folder_path = HOME+"/LPR_Image_temp"
Character_Model_Path = "character_recognition_model/saved_model.pb"

def receive_user_input(cursor):

    global input_type
    print(
        "\n\nWelcome! This app lets you to Recognise the License Plates from either several Videos or Images \n\n"
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
    print("close the plate figure to enter Recognition phase!!!... ")
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
        print("DISPLAY: NO PLATE FOUND!")
        return
    if num_images>3:
        Egxample_rate=int((num_images-1)/3)
        num_images=3
        THEREE_EXAMPLE_ID=[0,Egxample_rate,2*Egxample_rate]
    else:
        THEREE_EXAMPLE_ID=list(range(3))
    total_height = img_height * num_images
    fig, axs = plt.subplots(nrows=num_images, ncols=1, figsize=(2.5, total_height))

    # If there's only one image, axs is not a list
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, (_, row) in enumerate(df.iterrows()):
        if i in THEREE_EXAMPLE_ID:
            cropped_img = row['yoloobjectdetection.cropped_images']
    #         print(row)
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(cropped_img, -1, kernel)
            blended = cv2.addWeighted(cropped_img, alpha, sharpened, 1 - alpha, 0)
            denoised_image = cv2.medianBlur(blended, ksize=3)
    #         enlarged_img = cv2.resize(cropped_img, (300, 75), interpolation=cv2.INTER_LINEAR)
            axs[int(i/Egxample_rate)].imshow(denoised_image.astype(np.uint8))
            axs[int(i/Egxample_rate)].axis('off')  # hide axis

    plt.tight_layout()
    plt.show()


def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def get_dataset_partitions_tf(ds, num_batches, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_batches = int(train_split * num_batches)
    val_batches = int(val_split * num_batches)

    train_ds = ds.take(train_batches)
    val_ds = ds.skip(train_batches).take(val_batches)
    test_ds = ds.skip(train_batches).skip(val_batches)

    return train_ds, val_ds, test_ds

def character_recongition_model():
    ds=tf.keras.preprocessing.image_dataset_from_directory(
        "Iranis_Dataset_Files_resized",
        labels="inferred",
        label_mode='int',
        class_names=class_names,
        color_mode="rgb",
        batch_size=128,
        image_size=(30, 30),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False)
    print("loaded ds size:",len(ds))
    ds = ds.map(process)
    print(f"Actual dataset size: {len(list(ds))}")
    ds_train,ds_validation,ds_test=get_dataset_partitions_tf(ds,len(ds))
    print(f"Training samples: {len(ds_train)}")
    print(f"Validation samples: {len(ds_validation)}")
    print(f"Test samples: {len(ds_test)}")
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(int(83844*0.8))
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_validation = ds_validation.cache()
    ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(30, 30, 3)),
      tf.keras.layers.Dense(128,activation='relu'),
      tf.keras.layers.Dense(28)
    ])


    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_validation,
    )
    return model

def character_recongition(plate,model):
    # print(plate.shape)
    # plate_w,plate_h,dim=plate.shape
    # print(plate_w,"XX",plate_h)
    # plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    # plt.show()
    # time.sleep(5)
    global plate_checklist
    plate_checklist=[[True,1],[True,1],[True,2],[True,1],[True,1],[True,1],[True,1],[True,1]]
    try:
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    except:
        return ""
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # print(thresh)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    im2 = gray.copy()
    plate_num = ""
    # print(len(sorted_contours))
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 8:
            # print("continue1")
            continue

        if w / width > 0.4:
            # print("continue2")
            continue

        if w / width < 0.03:
            # print("continue3")
            continue
        # ratio = h / float(w)
        # # if height to width ratio is less than 1.5 skip
        # if ratio < 1.5: continue
        #
        # # if width is not wide enough relative to total width then skip
        # if width / float(w) > 15: continue

        area = h * w
        # print("x:",x," y:",y," w:",w," h:",h," area:",area)
        # if area is less than 100 pixels skip
        if area < 60:
            # print("continue4")
            continue

        # draw the rectangle
        if h / height < 0.3:
            y_min=int(max(0,y-(h/2)))
            y_max=int(min(height,y+(1.5*h)))
        else:
            y_min=y-10
            y_max=y+h+10
        # print("min",max(0,y-(h/2)))
        # print("max",min(height,y+(1.5*h)))
        rect = cv2.rectangle(im2, (x,y_min), (x+w,y_max), (0,255,0),2)
        # grab character region of image
        roi = thresh[y_min:y_max, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background

        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        # print("roi",roi)
        try:
            roi = cv2.medianBlur(roi, 5)
        except:
            # print("continue5")
            continue
        ready_input=cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
        img_array = keras.preprocessing.image.img_to_array(cv2.resize(ready_input, (30,30), interpolation = cv2.INTER_CUBIC))
        img_array = tf.expand_dims(img_array, 0)
        # plt.imshow(ready_input)
        # plt.show()
        # try:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        # print(
        #     "This image most likely belongs to {} ."
        #     .format(class_names[np.argmax(score)])
        # )
        text=class_names[np.argmax(score)]
        # print(text)
        if plate_check(text)==True:
            plate_num += text
        # except:
        #     print("failed")
        #     text = None
    # if plate_num != None:
    #     print("License Plate #: ", plate_num)
    return plate_num

def plate_check(text):
    flag=False
    req=None
    num=["0","1","2","3","4","5","6","7","8","9"]
    alpha=["A","B","D","Gh","H","J","L","M","N","P","PuV","PwD","Sad","Sin",
              "T","Taxi","V","Y"]
    plate_count=0
    for tp in plate_checklist:
        if tp[0]==True:
            req=tp[1]
            break
        plate_count=plate_count+1

    if req==1 and text in num:
        flag=True
        plate_checklist[plate_count][0]=False
    elif req==2 and text in alpha:
        flag=True
        plate_checklist[plate_count][0]=False

    if plate_count==3 and text=="0":
        flag=False
    return flag
# test_plate()
def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num

def Plate_Number_Recognition(df: pd.DataFrame, alpha = 0.7):
    global CARS_PLATES
    global sim_plates
    global plate_pic
    global flag
    num_images = len(df)
    if num_images==0:
        print("PLATE RECOGNITION: NO PLATE FOUND!")
        return
    for i, (_, row) in enumerate(df.iterrows()):
        cropped_img = row['yoloobjectdetection.cropped_images']
#         print(row)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(cropped_img, -1, kernel)
        blended = cv2.addWeighted(cropped_img, alpha, sharpened, 1 - alpha, 0)
        plate = cv2.medianBlur(blended, ksize=3)
        plate_num=character_recongition(plate,model)
        print(plate_num)
        if sum(c.isdigit() for c in plate_num)==7:
            if len(sim_plates)==0 :
                sim_plates.append(plate_num)
                # flag=True
                plate_pic=plate
            elif ((sim_plates[0][0]==plate_num[0] and sim_plates[0][1]==plate_num[1] and sim_plates[0][2]==plate_num[2]) or
            (sim_plates[0][1]==plate_num[1] and sim_plates[0][2]==plate_num[2] and sim_plates[0][3]==plate_num[3]) or
            (sim_plates[0][2]==plate_num[2] and sim_plates[0][3]==plate_num[3] and sim_plates[0][4]==plate_num[4]) or
            (sim_plates[0][3]==plate_num[3] and sim_plates[0][4]==plate_num[4] and sim_plates[0][5]==plate_num[5]) or
            (sim_plates[0][4]==plate_num[4] and sim_plates[0][5]==plate_num[5] and sim_plates[0][6]==plate_num[6]) or
            (sim_plates[0][5]==plate_num[5] and sim_plates[0][6]==plate_num[6] and sim_plates[0][7]==plate_num[7])):
                # print("sim:",sim_plates[0][0],"num:",plate_num[0])
                sim_plates.append(plate_num)
            else:
                # if not flag:
                if most_frequent(sim_plates)in CARS_PLATES:
                    plate_pic=[]
                    sim_plates=[]
                    continue
                CARS_PLATES.append(most_frequent(sim_plates))
                CARS_PLATES_PIC.append(plate_pic)
                plate_pic=[]
                sim_plates=[]
                # flag=False


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



def Character_Model_Load():
    global model
    if os.path.exists(Character_Model_Path):
        print("using pretrained character recognition model...")
        model = keras.models.load_model(model_path)
    else:
        model=character_recongition_model()
        model.save(model_path)


def main():
    cursor = evadb.connect().cursor()

    cursor = receive_user_input(cursor)

    cursor = Creating_LPR_YOLO_Function(cursor)

    result, cursor = YOLO_Predict_Function(cursor)

    Display_Plates(result)

    Character_Model_Load()

    Plate_Number_Recognition(result)

    print("Recognised plates:\n",CARS_PLATES)

    Del_Func(cursor)

    Clear_Func(video_folder_path)
    Clear_Func(image_folder_path)


if __name__ == "__main__":
    main()
