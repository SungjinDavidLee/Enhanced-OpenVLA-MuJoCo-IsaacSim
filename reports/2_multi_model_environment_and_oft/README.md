# 보고서 2. OpenVLA만으로 알 수 없던 것 — 여러 모델을 넣을 수 있는 평가 환경과 OpenVLA-OFT 시험 (준비 중)

OpenVLA는 카메라 한 장만 보고 매 순간 지시 하나를 내기 때문에, 실패가 화면을 잘못 보아서인지 움직임 때문인지 나누기 어렵다. 이 보고서는 손목 카메라와 로봇 자세 정보를 함께 쓰는 OpenVLA-OFT를 같은 평가 환경에 연결한 과정과, 그 모델의 시험 결과를 다룬다. GR00T · π0 계열 모델도 같은 환경에서 쓸 수 있도록 데이터 기록 방식을 맞춘 내용도 포함한다.

---

# Report 2. What OpenVLA alone could not show — an evaluation setup for several models, and OpenVLA-OFT (in preparation)

OpenVLA looks at a single camera image and issues one command at a time, so it is hard to tell whether a failure comes from perception or from motion. This report covers connecting OpenVLA-OFT, which also uses a wrist camera and the robot state, to the same evaluation setup, and its results. It also covers recording data so that GR00T and π0-family models can use the same setup.
