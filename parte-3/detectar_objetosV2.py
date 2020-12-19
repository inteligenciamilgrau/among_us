#https://github.com/sicara/tf2-yolov4
import tensorflow as tf

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import numpy as np

try:
    from PIL import ImageGrab, Image
except ImportError:
    import Image

import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

def plot_results(pil_img, boxes, scores, classes):
    plt.imshow(pil_img)
    ax = plt.gca()
    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0.3:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl % 6], linewidth=3))
            text = f'{CLASSES[cl]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    #plt.axis('off')
    plt.pause(0.00001)
    plt.show()
    plt.cla()

from timeit import default_timer as timer

if __name__ == '__main__':

    #HEIGHT, WIDTH = (640, 960)
    HEIGHT, WIDTH = (480, 640)
    #HEIGHT, WIDTH = (1280, 720)

    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=50,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.5,
    )

    model.load_weights("./yolov4.h5")
    model.summary()

    # COCO classes
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    plt.figure()
    plt.ion()

    printed_frames = 0
    sem_frame = 0
    start_time = timer()

    capturar = "grab"

    client = None
    cap = None

    faz_loop = True

    while faz_loop:
        image = None

        x = 00
        y = 125
        largura = 800
        altura = 600

        larguraFinal = largura + x
        alturaFinal = altura + y

        imagem_Pil = ImageGrab.grab([x, y, larguraFinal, alturaFinal])
        image = np.array(imagem_Pil)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = tf.image.resize(image, (HEIGHT, WIDTH))

            images = np.expand_dims(image, axis=0) / 255.0

            boxes, scores, classes, valid_detections = model.predict(images)

            partial_fps = int(printed_frames / (timer() - start_time))

            plot_results(
                images[0],
                boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
                scores[0],
                classes[0].astype(int),
            )

            printed_frames += 1

    cv2.destroyAllWindows()

