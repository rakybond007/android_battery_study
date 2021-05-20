# 호진 추가 learn model 설명

## 디렉토리 구조
### Scripts/logs/{실험idx}/{각 세팅에 해당하는 데이터 디렉토리}/
### setting : 해당 디렉토리의 실험이 진행된 세팅 값 저장. log_{idx} : 해당 세팅으로 실험된 결과, repeat idx

## 데이터 형태
### input.shape == (갯수, 4) -> 슬랙 참조 (세팅 값들)
### output.shape == (갯수, 16) -> 슬랙 참조 (전력값의 벡터들)

## CPU, wifi, bluetooth 로깅
### CPU : /data/local/tmp/log_cpu 
각 line : { key : 패키지명 value : 각 클러스터의 json array [ { key : speed index value : 해당 speed 에서 실행 시간 } ... ] }
### Wifi, bluetooth : /data/local/tmp/log_wifi_bluetooth
각 line : { key : 패키지명 value : json { key : 시간명 ex) idle_wifi, rx_wifi value : 실행 시간 } }
### 참고사항 : 값이 없을 경우에는, (실행시간이 0) wifi_bluetooth 로그에서는 빈 json 으로 존재할 수 있음

## 호진 실험 스크립트 설명
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