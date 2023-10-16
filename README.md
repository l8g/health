# rppg

heath monitoring



- [ ]  在处理PURE数据集时，PURE的帧率是60，但这里写成了30，需要搞清楚为什么

```python
hrv = get_hrv_label(raw_label, fs = 60.) # 这里原版是30，但是我觉得应该是60
```

