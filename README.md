# RL-atari

## DQN, CNN을 이용한 atari 강화학습 연습
- Try1 : Actor-critic, FullConnected, 단일 state로 강화학습 시도 으로 시도 -> 잘 동작하지 않아서 구글링 해본결과 DQN방식으로 예제가 많아서 DQN으로 선로 변경
- Try2 : DQN, FullConnected, 단일 State로 재시도 -> 실패, 구글링하여 deep-mind팀이 했던 하이퍼파라미터와 동작을 그대로 따라해보기로 변경 
- deep마인드 팀이 사용했던것을 정리해준 레퍼런스를 발견, -> 딥마인드팀이 사용했던소스는 아닌데.. 해당 레퍼런스가 어디있는지 까먹어서 https://keras.io/examples/rl/deep_q_network_breakout/ 이런 비슷한 느낌이였음  
기억나는 파라미터로는 replaybuffer 100만, 4개의 프레임을 state로 , 학습률 0.00025으로 100만번의 step이 진행되면 유의미한 결과가 나온다는 글이였으나  
내 환경에서는 그렇지 못했음  
어쨋든.. 해당 레퍼런스를 참고하여 자신의 소스에 맞춰서 수정
### **핵심은 4개의 프레임을 1개의 State로 사용하는것**  
- 아래와 같은 사진에서 공이 좌측인지 우측인지 상,하 정보같은게 없음 -> 그래서 추가적인 정보의 프레임이 필요하여 4개를 붙여 쓴다고 함
<img src="https://user-images.githubusercontent.com/57505385/224473114-9bf34af1-f681-41e7-adf4-78170c0b0280.png" width="300" height="300">  
- Try3: 4개의 프레임, FC, 단일 state로 연결하여 시도 -> 왜실패했었는지 기억이 나지 않으나.. 실패했었음 
<img src="https://user-images.githubusercontent.com/57505385/224473351-03b4f7b3-3c2a-46fd-a827-c66e7a60a1c8.png" width="300" height="300">  
- Try4 : DQN, CNN + FC, **4프레임을 State로 사용** -> 1만 episode 이후 Score가 7까지 올라갔으나 이후 정체됨

- Try4-1 : 해당 가중치를 가지고 human모드로 렌더링 해서 관찰하니 어느정도 학습이 이루어진형태로 보임  
잘 못치는 부분에서는 아직 학습이 덜 된것으로 추정됨 -> 이후 epsiolon을 30부터 시작하게 줄이고 decay를 60번 -> 200번으로 변경한후 재 학습  

- Try4-2 : 조금씩 학습이 다시 진행되는것을 확인  

- 평균 Score가 20정도 되었을때의 상태  

<img src="/sample.gif" width="300" height="300">
