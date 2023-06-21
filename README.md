# 2023 데이터 분석 캡스톤 디자인
경희대학교 2023학년도 1학기 데이터 분석 캡스톤 디자인 프로젝트 소스코드와 실험 내용

# 내용 요약
 stylegan2에 ada 기술과 freezeD를 적용하여 기존 이미지에 toonifying을 적용한다. ffhq 데이터셋으로 pretrain된 모델을 추가적으로 학습하여 커스텀 데이터셋으로 학습한 모델을 생성한다. pretrain 된 모델을 하나의 모델로 사용하고, fine tuning한 모델을 나머지 하나의 모델로 사용하여 두 모델을 network blending(layer swapping)하여 최종적인 결과물을 만드는 것을 목표로 한다.
 ada와 freezeD는 전이학습된 모델을 fine tuning하기 위해 사용되는 기술이다. 기본적인 stylegan2모델, ada를 적용한 모델, ada+freezeD를 적용한 모델이 각각 생성한 이미지를 이용해 fid score를 계산한다. 최종적으로 network blending을 이용해 생성된 이미지의 품질을 확인하는 것을 목표로 한다.
# Data set
## 1. 데이터 획득
![result](https://github.com/kth321/stylegan2_ada/assets/61428034/482db178-182c-4c50-8d77-368c2b6b48e4)

https://github.com/LynnHo/AttGAN-Cartoon-Tensorflow/tree/master

커스텀 데이터셋은 위의 github 리포지토리에서 제공하는 데이터를 이용하였다.이미지 파일의 경우 구글 드라이브에서 확인 가능하다.

## 2. pre-trained model
사전학습된 모델은 256x256 해상도의 ffhq 데이터셋을 사용하였다.
![download](https://github.com/kth321/stylegan2_ada/assets/61428034/0ea5eb9e-888e-44df-921a-d6ca9124b6c0)

## 3. 데이터 전처리
csv파일을 제거하고 이미지 파일은 원활한 학습을 위해 256x256 사이즈로 변환해준다
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/4027af12-734b-4c07-8424-cacf44e593d6)

