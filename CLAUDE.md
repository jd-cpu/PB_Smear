# 진단검사의학과 PB Smear 백혈구 탐지 웹앱 프로젝트 가이드

## 1. 페르소나 및 사용자 환경 (절대 준수)
- **사용자:** 코딩 비전공자이며 AI에 전적으로 의존하는 '바이브 코더'. 
- **환경:** Apple MacBook Air M2 (8GB RAM), macOS. 메모리가 매우 부족함.
- **제약:** 로컬 맥북에서는 절대 무거운 딥러닝 모델 학습을 진행하지 않는다. OOM(Out of Memory) 방지가 최우선이다.
- **응답 스타일:** 구구절절한 원리 설명보다는, '정확히 터미널에 쳐야 할 명령어'와 '복사해서 덮어씌울 수 있는 완성된 전체 코드 블록'을 직관적으로 제공하라.

## 2. 하이브리드 워크플로우 & 기술 스택
- **Phase 1. 모델 학습 (Google Colab용):** `train_colab.ipynb` 생성. `EfficientNetB0`를 활용한 전이학습 수행.
- **Phase 2. 객체 탐지 (OpenCV):** PB Smear 이미지 전체를 딥러닝으로 탐지하지 않는다. 연산 최소화를 위해 OpenCV의 HSV 색상 필터링(`cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`)을 사용하여 보라색/푸른색 계열(백혈구 핵)의 윤곽선을 찾고 Bounding Box를 치는 로직을 `app.py`에 구현한다.
- **Phase 3. 웹 서버 & UI (Flask + Bootstrap 5):** Flask를 이용해 `app.py`를 구축하고, `templates/index.html`에는 Bootstrap 5 CDN을 적용하여 모던하고 깔끔한 이미지 업로드/결과 확인 UI를 구성한다.
- **최종 제출:** 제공된 `Dockerfile`을 통해 패키징될 수 있도록 경로와 의존성을 맞춘다.

## 3. 코딩 규칙
- 코드는 PEP 8 스타일 가이드를 준수하며, 주요 로직(특히 OpenCV 필터링 값과 Flask 라우팅)에는 한글로 상세한 주석을 달아 사용자가 로직을 파악할 수 있게 한다.
- 패키지가 추가될 경우 항상 `requirements.txt`에 업데이트하는 명령어나 방식을 제시한다.