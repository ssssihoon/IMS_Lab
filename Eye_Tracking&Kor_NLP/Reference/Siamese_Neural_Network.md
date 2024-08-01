# Siamese Neural Network

~= (Twin Neural Network)

- 두 개의 서로 다른 입력 벡터에서 함께 작업하면서 동일한 가중치를 사용해 비교 가능한 출력 벡터를 계산하는 인공 신경망이다.

### 응용 분야

- 얼굴 인식

## 학습

- 삼중 손실 (Triplet Loss)
    - 구글에서 2015년 얼굴 인식 과제에서 소개한 Loss Function
    - Anchor-Positive 샘플의 거리를 최소화, Anchor-Negative 샘플의 거리를 최대화하는 것
- 대조적 손실 (Contrastive loss)
    - Positive 샘플이 비슷한(가까운) 표현으로 인코딩, Negative 샘플이 다른(더 먼) 표현으로 인코딩 되게 해 손실을 낮춘다.