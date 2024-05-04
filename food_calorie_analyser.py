from inference import InferencePipeline

from inference.core.interfaces.camera.entities import VideoFrame



# import opencv to display our annotated images

import cv2

# import supervision to help visualize our predictions

import supervision as sv



# create a simple box annotator to use in our custom sink

annotator = sv.BoxAnnotator()



def my_custom_sink(predictions: dict, video_frame: VideoFrame):

    # get the text labels for each prediction

    labels = [p["class"] for p in predictions["predictions"]]

    # load our predictions into the Supervision Detections api

    print("Predictions:", predictions)





    detections = sv.Detections.from_inference(predictions)

    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections),ac and the prediction labels

    image = annotator.annotate(

        scene=video_frame.image.copy(), detections=detections, labels=labels

    )

    # display the annotated image

    cv2.imshow("Predictions", image)

    cv2.waitKey(1)



pipeline = InferencePipeline.init(

    model_id="food_calorie/2",

    video_reference = r"C:\Users\Lenovo\Downloads\3378581-hd_1920_1080_25fps.mp4",

    on_prediction=my_custom_sink,

)



pipeline.start()

pipeline.join()