# 호진 추가 learn model 설명

## CPU, screen multiple linear regression : 각 실험 데이터를 이용해서 전력 소모량 선형회귀 학습
### CPU : 빅/리틀 코어의 frequency 지정 및 utilization을 순찾거으로 높여가며 연산 태스크 수행시키게 함으로써 전력 소모량 데이터 수집
파라미터 : 각 클러스터의 frequency, Utilization, 디바이스 온도, 디바이스 배터리 레벨
### Screen : 화면 구성 픽셀 전체를 특정 색상으로 채울 수 있도록 실험 세팅.
파라미터 : 화면 전체의 r, g, b 픽셀값의 평균값, 디바이스 온도, 디바이스 배터리 레벨

## Monitor_runtime* 실험 스크립트 설명
각 로그 파일 이름 기준 : 앱이름_실험번호_반복idx
(실험 번호는, 실험 세팅이나 목적 등이 달라져 이전 실험과 구분이 필요할 때 바꿔줌)
아직은 스크립트 실행마다 직접 실험번호와 반복 idx 를 직접 수정해가면서 실행.
모든 디바이스 관련 통신은 exec, exec_with_output 함수를 통해 함. adb -s {디바이스id} 에서 디바이스id는 직접 지정해야함.

### 앱 목록 : 다음웹툰 정주행기능, 네이버 웹툰, 네이버 컷툰, 스노우 필터 포함 촬영, 스크린 테스트 어플, 유투브 (방치, 자동재생), 트위치
참고 사항 : 네이버 웹툰, 컷툰, 트위치 등 실험은 특정 컨텐츠의 실행을 위해서 임의 시점 기준으로 해당 컨텐츠 실행까지의 버튼 입력을 구현해 놓은 것이라, 다른 영상을 재생하거나 다른 웹툰에 적용하는 등 응용을 위해서는 input tap 관련 수정이 필요

### 스크립트 내부 로깅 값 목록
1. util_logs : { cpu total util, gpu total util, cpu 클러스터당 util, cpu 클러스터당 freq, gpu greq, 해당줄의 로깅 start, end time }
2. cvt_logs : { 전류, 입력전류(충전중일때), 배터리레벨, 전압, 온도, 해당줄의 로깅 start, end time }
runtime_log~* 이름의 함수를 통해 로깅 스레드를 실행하며, 로깅 주기를 파라미터 값으로 넘겨줌.
