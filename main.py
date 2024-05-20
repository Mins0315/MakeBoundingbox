import os
import cv2 as cv

def convertToAbsoluteValues(size, box):
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])

    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1

    return (xIn, yIn, xEnd, yEnd)


def boundingBoxes(labelPath, imagePath):
    detections, groundtruths, classes = [], [], []

    # 모든 파일다 돌려본다.
    for labelfile in os.listdir(labelPath):
        # 파일중에서 .txt 확장자를 갖는 파일만 분석한다.
        if labelfile.endswith(".txt"):
            # os.path.splitext(labelfile) 해당 코드는 ( filename, 확장자) 형태의 튜플을 return 한다
            filename = os.path.splitext(labelfile)[0]

            # file을 디렉토리 + file_name으로 설정한다.
            with open(os.path.join(labelPath, labelfile)) as f:
                # 해당 파일에서 값들을 하나씩 읽어서 labelinfos []에 저장한다
                labelinfos = f.readlines()

            # .jpg 확장자를 갖는 파일만 경로를 저장한다.
            imgfilepath = os.path.join(imagePath, filename + ".jpg")
            # 이미지를 cv.imread() -> numpy.ndarray형태로 이미지를 변환해서 저장한다.
            img = cv.imread(imgfilepath)
            # 이미지의 크기와 높이, _ 채널( 사용 x) 를 저장한다.
            h, w, _ = img.shape


            for labelinfo in labelinfos:
                # label에서 공백을 제고 하고 스페이스바를 기준으로 문자열을 분리해서 parts에 넣는다.
                parts = labelinfo.strip().split()
                if len(parts) == 5:
                    # 라벨은 : 클래스명, 중앙값 x좌표, 중앙값 y 좌표, 넓이 , 높이
                    label, rx1, ry1, rw, rh = map(float, parts)
                    conf = 1.0  # 신뢰도 값이 없는 경우 기본값을 설정
                else:
                    raise ValueError(f"Unexpected label format: {labelinfo}")

                # YOLO 식 바운딩 박스 형식 -> VOC 바운딩 박스 형식
                x1, y1, x2, y2 = convertToAbsoluteValues((w, h), (rx1, ry1, rw, rh))
                # 파일정보에 [ 파일이름, 클래스명, 신뢰도, 바운딩 박스] 형식으로 만들어준다.
                boxinfo = [filename, label, conf, (x1, y1, x2, y2)]


                # 해당 라벨의 클래스가 클래스에 속해있지 않다면 새로운 클래스를 생성한다.
                if label not in classes:
                    classes.append(label)
                # 위에서 만든 파일정보 [ 파일이름, 클래스명, 신뢰도, 바운딩 박스] 을 groundTruth에 넣어준다.
                groundtruths.append(boxinfo)  # 모든 박스를 groundtruth로 처리

    classes = sorted(classes)
    return detections, groundtruths, classes

def drawBoundingBoxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        filename, label, conf, (x1, y1, x2, y2) = box
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label}: {conf:.2f}"
        cv.putText(image, label_text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def visualizeDetections(labelPath, imagePath, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # detections = [], groundtruhts = [ 파일이름, 클래스명, 신뢰도, 바운딩 박스] 정보들, classes는 [0,1,2,] 이런 정보들
    detections, groundtruths, classes = boundingBoxes(labelPath, imagePath)

    # 중복이 제거된 파일 이미지들의 리스트를 만들어준다.
    for filename in set([det[0] for det in groundtruths]):
        # 경로 + 파일이름 + .jpg로 이미지 파일들을 저장한다
        imgfilepath = os.path.join(imagePath, filename + ".jpg")
        # 이미지를 numpy.ndarray로 설정한다.
        img = cv.imread(imgfilepath)

        # groundtruhts에서 filename과 imagefiename이 동일한 image file을 찾아준다.
        groundtruth_boxes = [gt for gt in groundtruths if gt[0] == filename]

        drawBoundingBoxes(img, groundtruth_boxes, color=(255, 0, 0))  # Red for ground truths

        output_filepath = os.path.join(outputDir, filename + "_result.jpg")
        cv.imwrite(output_filepath, img)

# Example usage
labelPath = ' '
imagePath = ' '
outputDir = ' '

visualizeDetections(labelPath, imagePath, outputDir)
