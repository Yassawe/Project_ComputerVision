from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
from CNN import model
import cv2

img_path = "images/ex2.jpg"

a4_w = 210
a4_h = 297

webcam=True
crop=False
cap = cv2.VideoCapture(0)

def watershed(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (9, 9), cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(sure_fg, kernel, iterations=4)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    contours = cv2.watershed(img, markers)

    #TODO: костыль, надо элегантее решить
    contours[0, :] = 0
    contours[:, 0] = 0
    contours[-1, :] = 0
    contours[:, -1] = 0

    contours[contours != -1] = 0
    contours[contours == -1] = 255
    contours = contours.astype(np.uint8)
    #contours = cv2.dilate(contours, kernel, iterations=1) (хз оставить или нет)
    return contours


def get_cnts(image):
    edged=watershed(image)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas=[]
    if cnts:
        for c in cnts:
            areas.append(cv2.contourArea(c))
        max_index = areas.index(max(areas))
        return cnts, areas, max_index
    else:
        return cnts, areas, 0


def identify(img):
    img = cv2.resize(img, (200, 200))
    convlayers = [16, 32, 64, 64, 64]
    fc_layers = [512, 1]
    kernel = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    m_f = 2
    m_s = 2
    activation_hidden = 'relu'
    activation_last = 'sigmoid'
    lr = 0.01
    CNN_model = model(lr, convlayers, fc_layers, kernel, m_f, m_s, activation_hidden, activation_last)
    CNN_model.upload_weights('weights_stone.h5')
    return CNN_model.predict(img)


def crop_a4(image, box, A=210, B=297, scale=3, padding=50):

    (tl, tr, br, bl)=box
    box=[tl,tr,bl,br]

    p1 = np.float32(box)
    width = dist.euclidean((tl[0], tl[1]), (tr[0], tr[1]))
    height = dist.euclidean((tl[0], tl[1]), (bl[0], bl[1]))

    #если альбомная ориентация, то кропить на нее
    if width>height:
        A, B = B, A

    p2 = np.float32([[0, 0], [A*scale, 0], [0, B*scale], [A*scale, B*scale]])

    matrix = cv2.getPerspectiveTransform(p1, p2)
    cropped = cv2.warpPerspective(image, matrix, (A*scale, B*scale))
    cropped = cropped[padding:cropped.shape[0] - padding, padding:cropped.shape[1] - padding]
    return cropped

while True:
    #TODO: стримить сюда фрейм за фреймом с FPGA (или похуй с вебки покажем)
    if webcam:
        _, image = cap.read()
    else:
        image = cv2.imread(img_path)

    cnts, _, max_index = get_cnts(image)
    box = cv2.minAreaRect(cnts[max_index])
    box = np.array(cv2.boxPoints(box), dtype="int")
    box = perspective.order_points(box)

    if crop:
        image = crop_a4(image, box, a4_w, a4_h)
        longest_side = max(image.shape)

    else:
        (tl, tr, br, bl) = box
        w = dist.euclidean((tl[0], tl[1]), (tr[0], tr[1]))
        h = dist.euclidean((tl[0], tl[1]), (bl[0], bl[1]))
        longest_side = max([w, h])


    pixel_per_mm=longest_side/a4_h
    cnts, areas, _ = get_cnts(image)

    for i in range(len(cnts)):
        if cv2.contourArea(cnts[i]) < 100:
            continue

        box = cv2.minAreaRect(cnts[i])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        for (x, y) in box:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

        (tl, tr, br, bl) = box

        p_width = dist.euclidean((tl[0], tl[1]), (tr[0], tr[1]))
        p_height = dist.euclidean((tl[0], tl[1]), (bl[0], bl[1]))

        mm_width = (p_width / pixel_per_mm)
        mm_height = (p_height / pixel_per_mm)

        box_area_px = cv2.contourArea(box)
        contour_area_px = areas[i]

        if box_area_px !=0 :
            area_mm2=(contour_area_px/box_area_px)*(mm_height*mm_width)
        else:
            area_mm2=0


        #TODO: crop bounding box aroung the image
        # image = crop_somehow(image)

        if identify(image):
            cv2.drawContours(image, cnts[i], -1, (0, 255, 0), 3)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
            cv2.putText(image, "{:.1f}mm^2".format(area_mm2),
                        (int((tl[0] + bl[0] + tr[0] + br[0]) / 4), int((tl[1] + bl[1] + tr[1] + br[1])/4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    cv2.imshow("image1", image)
    cv2.waitKey(1)
