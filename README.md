# StyleGAN2 Finetuning&Toonify
경희대학교 2023-1학기 데이터 분석 캡스톤 디자인 프로젝트 

# 내용 요약
 stylegan2에 ada 기술과 freezeD를 적용하여 기존 이미지에 toonifying을 적용한다. ffhq 데이터셋으로 pretrain된 모델을 추가적으로 학습하여 커스텀 데이터셋으로 학습한 모델을 생성한다. pretrain 된 모델을 하나의 모델로 사용하고, fine tuning한 모델을 나머지 하나의 모델로 사용하여 두 모델을 network blending(layer swapping)하여 최종적인 결과물을 만드는 것을 목표로 한다.
 ada와 freezeD는 전이학습된 모델을 fine tuning하기 위해 사용되는 기술이다. 기본적인 stylegan2모델, ada를 적용한 모델, ada+freezeD를 적용한 모델이 각각 생성한 이미지를 이용해 fid score를 계산한다. 최종적으로 network blending을 이용해 생성된 이미지의 품질을 확인하는 것을 목표로 한다.
# Data set
## 1. 데이터 획득
![result](https://github.com/kth321/stylegan2_ada/assets/61428034/482db178-182c-4c50-8d77-368c2b6b48e4)

https://github.com/LynnHo/AttGAN-Cartoon-Tensorflow/tree/master

커스텀 데이터셋은 위의 github 리포지토리에서 제공하는 데이터를 이용하였다.이미지 파일의 경우 구글 드라이브에서 확인 가능하다.

## 2. pre-trained model
사전학습된 모델은 256x256 해상도의 ffhq 데이터셋을 사용하였다. (구글 드라이브 참고)
![download](https://github.com/kth321/stylegan2_ada/assets/61428034/0ea5eb9e-888e-44df-921a-d6ca9124b6c0)

## 3. 데이터 전처리
csv파일을 제거하고 이미지 파일은 원활한 학습을 위해 256x256 사이즈로 변환해준다
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/4027af12-734b-4c07-8424-cacf44e593d6)

# 모델 학습
## Training details
모델 학습 환경은 다음과 같다
- Google colab-pro
- V100 GPU
- 64-bit python==3.7 pythorch==1.7.1
- CUDA toolkit==11.0 or later
- Python libraries: pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

초기 training parameter 설정은 다음과 같다
- aug_strength=0.328
- train_count=10
- mirror_x=True
- mirror_y=False
- gamma_value=50.0
- augs='bg'
- config='auto'
- snapshot_count=1

## model training
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/3fdffe71-a4e5-4465-9560-50c21b37c9b7)
FFHQ 모델의 전이학습이 이루어지는 과정을 snapshot count마다 확인 가능

# 결과
## 1-1.이미지 결과물 확인 (stylegan2, stylegan2-ada, stylegan2-ada+FreezeD)
동일한 양의 데이터가 주어질 때, ada를 적용했을 때와 적용하지 않았을 때의 이미지 결과물을 비교한다.

시각적으로도 ada를 적용한 이미지가 일관성있고 안정적으로 이미지를 생성하는 것을 확인할 수 있다.
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/ea3746f8-ae42-4b7f-980d-f4f1d3471354)

추가로 ada와 Discriminator의 lower layer의 가중치를 고정시켜 학습하는 FreezeD를 적용해 이미지 결과물을 확인한다.

FreezeD를 적용하면 훈련 안정성과 generator의 자유도가 증가하며 모델의 학습 속도가 개선되어 안정적인 이미지 생성이 가능해진다.
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/80027973-fff5-4f01-93d9-037880169a95)

각 모델의 FID 확인 결과, ADA+FreezeD를 적용한 모델이 ADA만 적용한 모델보다 낮은 FID 점수를 보이는 것을 확인할 수 있다.
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/3e968269-076d-49b1-b8df-42880f32c5d5)

마지막으로 Layer swapping을 적용해 FFHQ 모델과 pre trained된 모델을 합친 결과물을 확인한다.

layer swapping을 사용하면 낮은 해상도의 레이어에서 이미지의 구조를, 높은 해상도의 레이어에서 이미지의 스타일을 얻을 수 있다. 여기서는 4x4, 8x8, 16x16, 32x32, 64x64, 128x128 layer에서 각각 fine tuning한 모델을 합성하는 실험을 하였다.
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/80a6ab33-660c-4b56-b6fc-53272d2f1cdd)
![image](https://github.com/kth321/stylegan2_ada/assets/61428034/a3615277-156f-4143-816c-77ebac68d82e)
128x128 layer에서 두 모델의 layer swapping 결과가 가장 좋은 것을 확인할 수 있다.

<hr>
# 결과
 해당 프로젝트로 구축된 StyleGAN2-ADA 모델은 StyleGAN2-ADA는 생성된 이미지의 품질과 다 양성을 향상시키는데 사용될 수 있을 것이다. Cartoon 이미지에 대해 학습된 모델은 더욱 현실적이고 정교한 Cartoon 스타일의 이미지를 생성할 수 있다. 따라서 고품질의 Cartoon 이미지를 생성할 수 있을 것이라 기대된다.

 # Reference
 - StyleGAN2를 활용한 스케치 기반 애니메이션 이미지 생성(https://www-dbpia-co-kr-ssl.webgate.khu.ac.kr/journal/articleDetail?nodeId=NODE11224229)
- 전이학습이란(Transfer Learning)?(https://dacon.io/forum/405988)
- Analyzing and Improving the Image Quality of StyleGAN(https://arxiv.org/abs/1912.04958)
- FID(https://pubs.acs.org/doi/10.1021/acs.jcim.8b00234)
- Cartoon-Stylegan2::Stylegan으로 내 얼굴 만화 캐릭터 만들기(https://happy-jihye.github.io/gan/gan-21/)
