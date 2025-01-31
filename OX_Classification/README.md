# [O_X 분류기를 이용해 직접 그림을 그려 분류하기](https://github.com/ssssihoon/IMS_Lab/blob/main/OX_Classification/OX_Drawing/OX_Drawing.ipynb)


## <O_X 클래스 분류>

OX 그림을 직접그려 O와 X를 분류하는 CNN모델 만들기 및 CAM으로 가중치 분석

..2주간의 작업

직접 구현과 gpt를 사용해 가이드라인과 전처리 과정을 따왔다.

### = 전처리 과정

- jpg, png로 두 확장자
    - 이미지가 jpg, png로 두 확장자로 구성이 되어있는 데이터여서 이를 어떻게 순차적으로(1.jpg, 2.png, 3.jpg ```n.png) 데이터셋에 넣는 부분에서 복잡했다.
- 이미지o, x에 해당하는 레이블로 만드는 것
    - 그에따른 원-핫 인코딩을 생각했지만 다른 생각이 들었는지 O : 1, 나머지 : 0 이렇게 단일 클래스 분류를 해버렸다. → 다시 원-핫 인코딩 라이브러리를 사용해 해결
- 이미지(2차원 데이터)를 한 블럭으로 생각해서 그것을 러닝 시키는 것
    - 이미지는 행과 열로 구성되어있고 그에 해당하는 개수(Layer)를 돌아가면서 fold검증을 하려 했으나 머리속에서는 그렇게 하길 원하는데 코드를 작성하는 부분에 있어서 어떻게 해야하는 지 막막해 개-고양이 분류하는 코드를 검색해 해결했다.

### = 모델링

- 분류가 거의 되지 않았음. 정확도 60% 정도
    - 정상적으로 O, X가 그려진 이미지가 분류가 되지 않았다. 에폭에 따른 정확도가 금방 일정 값에 수렴하고 Loss가 내려가지도 않았다. 그래서 커널의 크기를 증가시켰다. (3, 3) → (4, 4) 그에 따른 Pooling_size도 증가시켰다.
- 그래도 분류가 70% 정도로 잘 되지 않음
    - 데이터가 간단하지만 특징들이 복잡할 것이라 생각해 연산의 횟수를 늘리기 위해 레이어를 더 쌓고, 뉴런의 수도 증가시켰다.
    - 과하게 층을 쌓거나 뉴런의 수를 증가시키니 70언저리로 나왔고, 적당하게 바꾸니 80%초반까지 나오게 됐다.
- OX분류를 당연히 잘 할 것이라고 예상을 했는데 분류된 이미지를 살펴보면 당연히 분류 되어야 한다고 생각하는 것들(정상적인 O, X)는 어느정도 분류가 됐지만 , 분류가 안될 것 같은 것도 당연하게 분류가 되지 않음
    - 그 속에서 구석진 부분에 그려져있는 ox가 있었음.
    - 패딩을 생각하지 못했다.
        - 패딩을 적용해 러닝을 돌렸지만 정확도가 크게 높아지거나 하진 않았다. 살짝은 높아졌다.(1.25%)


### = CAM

- 모델을 저장해 가중치 영역을 이미지에 투영해서 색상으로 나타내 결과를 보았다.
    - 따뜻한 색상 (빨 ~ 노) : 가중치를 많이 받은 영역
    - 차가운 색상 (파 ~ 초) : 가중치를 적게 받은 영역
    - 보라색 : 가중치를 많이도 적게도 받지 않은 영역
        - O 의 경우 : 360’로 보았을 때, 0’~45’ ~ 200’~220’ 부근에서 가중치 를 많이 받았다.
        - X 의 경우 : 음의 기울기 선은 가중치를 받은 반면, 양의 기울기 선은 가중치를 적게 받았다.

→ 구석에 O, X 가 그려진 이미지를 패딩을 했음에도 분류를 못하는 것을 보고 데이터셋이 너무 적었다고 생각이 든다. 왜냐하면 구석의 O, X는 결국 중앙에 있다고 하면 다른 정상적인 O, X와 다를 것이 없는 데이터이기 때문이다. 데이터 양이 많았다면 훨씬 더 잘 분류를 했을 것이다.

그리고 미지수 x처럼 적은 X가 있는데 이것은 중앙이 O로 보여질 수도 있다고 생각이 들어 이마저도 데이터양이 많다면 분류가 가능할 것 같다는 생각이 든다.
