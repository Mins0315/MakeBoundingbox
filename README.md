# bounding box 만들기
# 해당 코드는 Obejct Detection 코드에서 모델을 평가하는 mAP 구현을 하고 있습니다.
# kaggle에서 https://www.kaggle.com/datasets/biancaferreira/african-wildlife 해당 코드를 가지고 작성하였습니다.
# https://herbwood.tistory.com/3 자료를 활용하여 제작하였습니다.


# 이미지에서 바운딩 박스 제작 방식에는 크게 2가지가 존재합니다.
# 1. VOC 방법 좌상단 좌표 + 넓이와 높이가 제공 된다.
# 2. YOLO 방법 중앙 좌표와 넓이와 높이가 주어진다.


# 해당 코드는 VOC 좌표를 기준으로 만었으나, 해당 데이터는 YOLO 바운딩 박스 형식을 사용하고 있다.
