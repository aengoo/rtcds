# rtcds
real-time criminal detection system

　

This project has no commercial purpose.

I'd appreciate to any advice for my ignorance of the license notation.

mail : nu_start@naver.com

　

------

　

### Evaluation History

Default Option Table

| det_res | idt_res | targets | bbox_pad(nHD/VGA) | det_backbone | H/W  | OS   |
| ------- | ------- | ------- | ----------------- | ------------ | ---- | ---- |
|         |         |         |                   |              |      |      |

|             | Predict ▶ |      |      |      |              |
| ----------- | --------- | ---- | ---- | ---- | ------------ |
| **Truth ▼** | CHOI      | KANG | KIM  | LIM  | UNIDENTIFIED |
| CHOI        |           |      |      |      |              |
| KANG        |           |      |      |      |              |
| KIM         |           |      |      |      |              |
| LIM         |           |      |      |      |              |

　

------

　

| det_res | idt_res | targets | bbox_pad     | det_backbone | H/W    | OS   |
| ------- | ------- | ------- | ------------ | ------------ | ------ | ---- |
| nHD     | FHD     | 4       | 10 (nHD/VGA) | ResNet-50    | 1080TI | Win  |

```
self.tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
```

　

Dataset : beta-set

**Test1. No Tracking**

|             | Predict ▶ |      |      |      |              |
| ----------- | --------- | ---- | ---- | ---- | ------------ |
| **Truth ▼** | CHOI      | KANG | KIM  | LIM  | UNIDENTIFIED |
| CHOI        | 604       | 2    | 16   | 8    | 518          |
| KANG        | 0         | 611  | 2    | 2    | 111          |
| KIM         | 3         | 0    | 499  | 0    | 546          |
| LIM         | 0         | 0    | 0    | 387  | 842          |

Overall Acc : 0.506

Overall Time : 33ms

　

**Test2. Tracking**

|             | Predict ▶ |      |      |      |              |
| ----------- | --------- | ---- | ---- | ---- | ------------ |
| **Truth ▼** | CHOI      | KANG | KIM  | LIM  | UNIDENTIFIED |
| CHOI        | 768       | 0    | 58   | 0    | 322          |
| KANG        | 0         | 723  | 0    | 0    | 3            |
| KIM         | 3         | 0    | 877  | 0    | 171          |
| LIM         | 0         | 0    | 0    | 911  | 238          |

Overall Acc : 0.809

Overall Time : 33ms

　

**Test3. Counted at endpoint of tracking**

|             | Predict ▶ |      |      |      |              |
| ----------- | --------- | ---- | ---- | ---- | ------------ |
| **Truth ▼** | CHOI      | KANG | KIM  | LIM  | UNIDENTIFIED |
| CHOI        | 2         | 0    | 1    | 0    | 1            |
| KANG        | 0         | 3    | 0    | 0    | 0            |
| KIM         | 0         | 0    | 5    | 0    | 0            |
| LIM         | 0         | 0    | 0    | 3    | 3            |

Overall Acc : 0.722

Overall Time : 34ms

　

**Test4. Tracking, Landmark-68 (default is 5)**

|             | Predict ▶ |      |      |      |              |
| ----------- | --------- | ---- | ---- | ---- | ------------ |
| **Truth ▼** | CHOI      | KANG | KIM  | LIM  | UNIDENTIFIED |
| CHOI        | 760       | 0    | 68   | 0    | 320          |
| KANG        | 0         | 723  | 0    | 0    | 3            |
| KIM         | 0         | 0    | 877  | 0    | 171          |
| LIM         | 0         | 8    | 0    | 405  | 816          |

Overall Acc : 0.666

Overall Time : 33ms

　

**Test5. No Tracking, Landmark-68 (default is 5)**

|             | Predict ▶ |      |      |      |              |
| ----------- | --------- | ---- | ---- | ---- | ------------ |
| **Truth ▼** | CHOI      | KANG | KIM  | LIM  | UNIDENTIFIED |
| CHOI        | 604       | 0    | 17   | 9    | 518          |
| KANG        | 0         | 611  | 2    | 0    | 113          |
| KIM         | 4         | 0    | 527  | 0    | 517          |
| LIM         | 0         | 4    | 0    | 747  | 478          |

Overall Acc : 0.599

Overall Time : 33ms

　
