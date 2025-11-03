# CIFAR-10 다중 클래스 분류를 위한 MLP 모델의 하이퍼파라미터 최적화 알고리즘

## 실행
```
python auto.py
```

## 파이프라인
<p align="center"> <img src="pipeline.png" alt="pipeline" width="85%"> </p>

## 결과
### 신뢰도 75%
<table>
<tr>
  <th>Run #1 (2025-11-03 03:57:26)</th>
  <th>Run #2 (2025-11-03 05:55:27)</th>
</tr>
<tr>
  <td align="center">
    <img src="./final/learning_curves_75_20251103_035726.png" width="49%">
    <img src="./final/learning_curves_acc_75_20251103_035726.png" width="49%"><br>
    <sub>Loss</sub>&nbsp;&nbsp;&nbsp;&nbsp;<sub>Accuracy</sub>
  </td>
  <td align="center">
    <img src="./final/learning_curves_75_20251103_055527.png" width="49%">
    <img src="./final/learning_curves_acc_75_20251103_055527.png" width="49%"><br>
    <sub>Loss</sub>&nbsp;&nbsp;&nbsp;&nbsp;<sub>Accuracy</sub>
  </td>
</tr>
</table>

---

### 신뢰도 85%
<table>
<tr>
  <th>Run #1 (2025-11-03 03:04:19)</th>
  <th>Run #2 (2025-11-03 05:02:49)</th>
</tr>
<tr>
  <td align="center">
    <img src="./final/learning_curves_85_20251103_030419.png" width="49%">
    <img src="./final/learning_curves_acc_85_20251103_030419.png" width="49%"><br>
    <sub>Loss</sub>&nbsp;&nbsp;&nbsp;&nbsp;<sub>Accuracy</sub>
  </td>
  <td align="center">
    <img src="./final/learning_curves_85_20251103_050249.png" width="49%">
    <img src="./final/learning_curves_acc_85_20251103_050249.png" width="49%"><br>
    <sub>Loss</sub>&nbsp;&nbsp;&nbsp;&nbsp;<sub>Accuracy</sub>
  </td>
</tr>
</table>

---

### 신뢰도 95%
<table>
<tr>
  <th>Run #1 (2025-11-03 02:24:58)</th>
  <th>Run #2 (2025-11-03 04:23:24)</th>
</tr>
<tr>
  <td align="center">
    <img src="./final/learning_curves_95_20251103_022458.png" width="49%">
    <img src="./final/learning_curves_acc_95_20251103_022458.png" width="49%"><br>
    <sub>Loss</sub>&nbsp;&nbsp;&nbsp;&nbsp;<sub>Accuracy</sub>
  </td>
  <td align="center">
    <img src="./final/learning_curves_95_20251103_042324.png" width="49%">
    <img src="./final/learning_curves_acc_95_20251103_042324.png" width="49%"><br>
    <sub>Loss</sub>&nbsp;&nbsp;&nbsp;&nbsp;<sub>Accuracy</sub>
  </td>
</tr>
</table>
