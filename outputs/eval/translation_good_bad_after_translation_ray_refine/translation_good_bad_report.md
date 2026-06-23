# Translation Good/Bad Frame Report

- Input CSV: `/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine/all_frame_person_translation_rows.csv`
- Bad threshold: `0.500m`
- Severe threshold: `0.800m`
- Frames: `7619`
- Frame-person rows: `21852`

## BASE

| level | count | mean | p50 | p90 | p95 | p99 | max | >bad | >severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 21852 | 0.073031 | 0.066415 | 0.122165 | 0.143501 | 0.201395 | 1.026707 | 5 | 3 |
| frame max | 7619 | 0.103565 | 0.096115 | 0.153807 | 0.177242 | 0.241184 | 1.026707 | 5 | 3 |

Good frames: `7614` / `7619`

## HSI

| level | count | mean | p50 | p90 | p95 | p99 | max | >bad | >severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 21852 | 0.091388 | 0.081698 | 0.158595 | 0.184639 | 0.243066 | 1.079681 | 5 | 2 |
| frame max | 7619 | 0.135879 | 0.129948 | 0.196517 | 0.220695 | 0.283825 | 1.079681 | 5 | 2 |

Good frames: `7614` / `7619`

## Base vs HSI

- HSI better persons: `8385` / `21852`
- HSI worse by >5cm persons: `5174`
- Bad-to-good rescued persons: `1`
- Newly bad persons: `1`
- Both bad persons: `4`
- Bad-to-good rescued frames: `1`
- Newly bad frames: `1`
