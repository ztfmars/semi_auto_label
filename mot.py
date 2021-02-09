# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from xml_save import annotation_single_img
from pathlib import Path

from PIL import Image

select_classes = {
    1:"car",
    2:"truck",
    3:"jeep"
}

input_cmd = "Enter differ num for type: %s"%(str(select_classes))
lost_warning = "Objs are lost! Please Relabel again after 3 s! "


# args = {
#     "interval" :10,
#     "pic_dir": r"D:\2WORK\Project\Self_Label\semi_auto_label\mot\JPEGImages",
#     "xml_dir": r"D:\2WORK\Project\Self_Label\semi_auto_label\mot\Annotations",
#     "set_width":1080,
#     "set_height":600,
#     "video" : "drone.mp4",
#     "tracker" :"mosse"
# }


def mot_detect(args):
    def nothing(emp):
        pass

    print("----------> create the trackers")
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(args["tracker"].upper())
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        trackers = cv2.MultiTracker_create()
    # tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None
    tclass = []
    start_time = time.time()
    failure_flag = False


    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])
    # initialize the FPS throughput estimator
    fps = None

    frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    loop_flag = 0
    pos = 0

    print("----------> read the frames")
    cv2.namedWindow('Frame')
    cv2.createTrackbar('time', 'Frame', 0, frames, nothing)


    # loop over frames from the video stream
    while True:
        # process-bar setting
        if loop_flag == pos:
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', 'Frame', loop_flag)
        else:
            pos = cv2.getTrackbarPos('time', 'Frame')
            loop_flag = pos
            vs.set(cv2.CAP_PROP_POS_FRAMES, pos)

        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=int(args["set_width"]),
                                                height=int(args["set_height"]))
        frame0= frame.copy()
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            print("----------> update the trackers' roi area")
            # grab the new bounding box coordinates of the object
            (success, boxes) = trackers.update(frame)
            print("[INFO] success / box num", success, len(boxes))

            # check to see if the tracking was a success
            if success:
                tpos = []
                for num, box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]
                    new_pos = [x,y, x+w, y+h]
                    tpos.append(new_pos)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, tclass[num], (x, y- 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                now_time = time.time()
                if now_time -start_time > float(args["interval"]) and len(boxes)>0:
                    start_time= now_time
                    pic_name = str(time.time())+".jpg"
                    pic_fullpath = Path(args["pic_dir"]).joinpath(pic_name)
                    print("[INFO] save new pic:", pic_fullpath)
                    r = Image.fromarray(frame0[:, :, 2]).convert('L')
                    g = Image.fromarray(frame0[:, :, 1]).convert('L')
                    b = Image.fromarray(frame0[:, :, 0]).convert('L')
                    img = Image.merge("RGB", (r, g, b))
                    print("----------> save the pic and annotation xml")
                    img.save(pic_fullpath)
                    annotation_single_img(args["pic_dir"], pic_name,
                                                                args["xml_dir"], tclass, tpos)
                # update the FPS counter
                fps.update()
                fps.stop()
                # initialize the set of information we'll be displaying on
                # the frame
                info = [
                    ("Tracker", args["tracker"]),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps())),
                ]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                trackers.clear()
                trackers = cv2.MultiTracker_create()
                failure_flag = True
                initBB = None
                tclass = []
                cv2.putText(frame, lost_warning, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame)
        if failure_flag:
            cv2.waitKey(5000) & 0xFF
            failure_flag = False
        else:
            key = cv2.waitKey(100) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            print("----------> select roi by mouse")
            cv2.putText(frame, input_cmd, (10,  20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            # show the classes choice
            ckey = cv2.waitKey(0) & 0xFF
            if int(ckey-48) > len(select_classes) or int(ckey-48) <=0:
                continue
            cname = select_classes[int(ckey-48)]
            print("[INFO] choose type to label:", cname)
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROIs("Frame", frame, fromCenter=False,
                                   showCrosshair=True)

            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            initBB = tuple(map(tuple, initBB))
            if str(initBB) == '()':
                print("[WARNING] There is no select ROIs!")
                # initBB==None
            else:
                for bb in initBB:
                    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                    trackers.add(tracker, frame, bb)
                    tclass.append(cname)


            fps = FPS().start()

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            print("----------> quit all process")
            break

        elif key == ord("r"):
            print("----------> clear all roi trackers")
            tclass = []
            trackers.clear()
            trackers = cv2.MultiTracker_create()
            cv2.putText(frame, input_cmd, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            # show the classes choice
            ckey = cv2.waitKey(0) & 0xFF
            cname = select_classes[int(ckey - 48)]
            print("[INFO]You have chosen the class:%s"%cname)
            initBB = cv2.selectROIs("Frame", frame, fromCenter=False,
                                 showCrosshair=True)
            initBB = tuple(map(tuple, initBB))
            if str(initBB) == '()':
                print("[WARNING] There is no select ROIs!")
                # initBB==None
            else:
                print("---------->add new roi trackers")
                for bb in initBB:
                    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                    trackers.add(tracker, frame, bb)
                    tclass.append(cname)
        # else:
        #     continue

    # if we are using a webcam, release the pointer
    if not args.get("video", False):
        vs.stop()
    # otherwise, release the file pointer
    else:
        vs.release()
    # close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mot_detect(args)