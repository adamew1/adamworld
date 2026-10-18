# BENCHMARKS.md

Device: **cpu**  ·  backend: **mock**

> On CPU the mock backend runs and VRAM reads 0. Re-run on a T4/P100 with the
> `[gpu]` extra for real default-backend numbers.

## resolution 256
```
stage           time(s)   peak VRAM(MB)
---------------------------------------
text_to_image      0.07            0.0
panorama           0.01            0.0
depth              0.00            0.0
lift               0.01            0.0
---------------------------------------
TOTAL              0.09            0.0
```

## resolution 512
```
stage           time(s)   peak VRAM(MB)
---------------------------------------
text_to_image      0.13            0.0
panorama           0.02            0.0
depth              0.00            0.0
lift               0.05            0.0
---------------------------------------
TOTAL              0.20            0.0
```
