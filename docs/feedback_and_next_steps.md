# 자문 피드백과 다음 단계

TDA 전공 교수님께 자문을 구하고 받은 피드백과, 그에 따라 이어서 해볼 작업을 정리한 문서입니다.
프로젝트를 다시 시작할 때 이 문서부터 봅니다.

- 면담일: 2026년 10월 8일(목)
- 상태: 보류 (당장 진행하지 않음)

---

## 1. 보낸 질문

- TDA 피처 일부(`TDA_L1_H0`)가 20일 변동성과 상관 0.81 → 예측이 나아지지 않음. 변동성만 넣은 모델보다도 낫지 않음.
- 노름이 점구름 크기(RMS 쌍거리)의 k제곱으로 커진다고 보고 크기^k로 나누니 상관이 0.13으로 떨어짐
  → 피처가 점구름의 **모양보다 크기**를 재고 있었던 것으로 판단 ([`02_tda_features.ipynb`](../notebooks/02_tda_features.ipynb) §2-1).
- 원 논문(Gidea & Katz, 2018)에서는 위기 전 노름 증가 자체가 결과였는데, **크기를 빼는 게 맞는 방향인지,
  맞다면 크기로 나누는 것 말고 어떤 방법이 있는지** 질문.

## 2. 받은 피드백 (면담 직후 기억한 내용)

1. **H₀보다 H₁에 집중하라.** 금융 시계열이므로 연결성분(H₀)보다 주기·루프를 띠는 H₁이 더 중요하다. H₁ 해석에 집중할 것.
2. **persistence landscape는 원래 이런 용도로 만든 게 아니다.** 한 데이터 군집과 다른 군집이 비슷한지 보려고,
   즉 landscape의 평균·분산 등을 계산하려고 만든 도구다. 지금처럼 쓰면 별 쓸모가 없을 수도 있다.
3. **쓸모없는 피처는 아예 빼고 다시 돌려봐라.** 예: 변동성과 상관 0.81이던 `TDA_L1_H0` 제외.
4. **더 나아가고 싶다면 extended persistence로 확장해 보라.** 지금 쓰는 필트레이션은 앞 집합이 다음 집합에
   포함되는 한 방향(⊆)이지만, 끝까지 간 뒤 반대 축, 즉 앞 집합이 다음 집합을 포함하는(⊇) 방향을 이어 붙일 수 있다.
5. **데이터 자체의 구조도 봐야 한다.** TDA 결과를 해석하기 전에 점구름이 원래 어떻게 생겼는지부터 볼 것.

## 3. 해석과 확인할 점

> 이 절은 피드백을 바탕으로 **우리가 덧붙인 해석**입니다. 교수님 말씀(2절)과 구분해서 읽을 것.

**1번 (H₁)**
- 우리 결과와 방향이 같다. H₀ 노름은 MST 간선 길이, 즉 크기의 함수다(§2-1 §F). H₁은 크기에 반응하는 기울기가
  H₀보다 작고(§E: MST 기준 L¹ 1.46 vs 2.05), 크기^k로 나눈 뒤 변동성과 상관 0.045(§D).
- 확인할 점: 지금 점구름은 하루 = 4개 지수 수익률로 된 4차원 점 하나라서, 점구름 안에 시간 순서가 없다.
  여기서 나온 H₁ 루프가 시간적 주기를 뜻하는지는 따로 따져봐야 한다.
  시계열 주기를 H₁으로 잡는 표준 방법은 지연 임베딩(sliding window embedding; Perea & Harer, 2015)이다.

**2번 (landscape)**
- Bubenik(2015)이 landscape를 만든 목적: 다이어그램은 평균을 낼 수 없지만 landscape는 함수라서
  집단별 평균 landscape를 구하고 두 집단을 검정할 수 있다.
- 원래 용도에 맞는 활용 예: 위기 직전 윈도우들의 평균 landscape와 평상시 평균 landscape를 순열 검정으로 비교.
  매일 노름을 숫자 하나로 뽑아 예측 피처로 넣는 지금 방식과는 다른 질문("위기 전후로 위상 구조가 다른가")이다.

**3번 (피처 제거)**
- 이미 있는 실험 ⑥(저중복 TDA)은 `L1_H0`·`L2_H0`·`L1_H1_std20`을 뺐지만 **TDA만 단독**으로 쓴 실험이다.
  "기본 22 + TDA"에서 빼고 돌린 실험은 아직 없다.

**4번 (extended persistence)**
- Cohen-Steiner, Edelsbrunner & Harer (2009). 아래에서 위로(sublevel set) 쓸어 올린 뒤
  위에서 아래로(superlevel set) 다시 내려오는 구간을 이어 붙여, 무한 막대 없이 모든 막대를 유한하게 만든다.
- 포함 방향이 섞인 필트레이션 일반은 zigzag persistence(Carlsson & de Silva, 2010)이며,
  extended persistence는 levelset zigzag와 정보가 같다(Carlsson, de Silva & Morozov, 2009).
- 확인할 점: extended persistence는 보통 공간 위의 함수(sublevel/superlevel)에 대해 정의된다.
  지금의 VR 점구름에 바로 붙이기보다 수익률 시계열을 경로 그래프 위의 함수로 보는 등 설정을 바꿔야 한다.

**5번 (데이터 자체의 구조)**
- 아래는 점구름의 원래 모양을 볼 때 우리 데이터에서 짚어볼 만한 것들이다.
- 4개 지수는 서로 상관이 높아서 4차원 점구름이 실제로는 한두 방향으로 길쭉하게 몰려 있을 수 있다.
  그렇다면 "4차원 점구름의 루프"가 무엇을 뜻하는지가 달라진다 → PCA 설명분산 비율 확인 (viz4에 2D 투영만 있음).
- 수익률은 꼬리가 두껍고, 변동성이 몰려서 나타난다(volatility clustering). 점구름 크기가 윈도우마다
  최대 약 11배 차이 나는 것(§2-1 §C)도 이 때문이다. 극단적인 날 몇 개가 다이어그램을 좌우하는지 확인할 것.
- 시간 순서: 1번에서 적었듯 지금 점구름에는 시간 정보가 없다. 데이터가 원래 시계열이라는 구조를
  TDA 입력에 어떻게 담을지(예: 지연 임베딩)도 이 질문의 일부일 수 있다.

## 4. 다음에 할 일

**우선 — 피처 제거 실험** ([`common_eval.py`](../notebooks/common_eval.py)의 `EXPERIMENTS`에 추가)

| 실험 | 피처 | 묻는 것 | 근거 |
|---|---|---|---|
| ③-a | 기본 22 + TDA − `L1_H0` | 교수님 예시 그대로 | 피드백 3 |
| ③-b | 기본 22 + H₁ 피처만 (`L1_H1`, `L2_H1`, `diff1`, `std20`) | H₁만으로 기여가 있는가 | 피드백 1 |
| ③-c | 기본 22 + H₁ 피처 ÷ 크기^k | 크기를 뺀 H₁ 모양 정보의 기여 | 피드백 1 + 원래 질문 |

**그다음**
- [ ] 데이터 구조 점검: 4개 지수 수익률의 PCA 설명분산, 꼬리·이상치가 다이어그램에 미치는 영향, 시기별 점구름 모양 비교
- [ ] H₁ 해석: 위기 구간과 평상시 H₁ 바코드 비교, 루프가 무엇에 대응하는지 확인
- [ ] landscape를 원래 용도로: 위기 전 vs 평상시 평균 landscape 순열 검정
- [ ] (선택) extended persistence 공부 후 적용 방식 설계

## 참고문헌

- Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *JMLR*, 16, 77–102.
- Carlsson, G., & de Silva, V. (2010). Zigzag persistence. *Foundations of Computational Mathematics*, 10(4), 367–405.
- Carlsson, G., de Silva, V., & Morozov, D. (2009). Zigzag persistent homology and real-valued functions. *SoCG*.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2009). Extending persistence using Poincaré and Lefschetz duality.
  *Foundations of Computational Mathematics*, 9(1), 79–103.
- Perea, J. A., & Harer, J. (2015). Sliding windows and persistence: An application of topological methods to signal analysis.
  *Foundations of Computational Mathematics*, 15(3), 799–838.
