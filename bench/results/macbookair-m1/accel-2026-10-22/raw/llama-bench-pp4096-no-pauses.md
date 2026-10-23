| model                          |       size |     params | backend    | threads | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | MTL,BLAS   |       4 |      256 |   0 |          pp4096 |        317.03 ± 0.19 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | MTL,BLAS   |       4 |      256 |   1 |          pp4096 |        284.58 ± 0.60 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | MTL,BLAS   |       4 |      512 |   0 |          pp4096 |        282.29 ± 7.90 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | MTL,BLAS   |       4 |      512 |   1 |          pp4096 |        251.36 ± 3.25 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | MTL,BLAS   |       4 |     1024 |   0 |          pp4096 |        245.17 ± 1.05 |
| gemma4 E4B Q4_0                |   3.91 GiB |     7.46 B | MTL,BLAS   |       4 |     1024 |   1 |          pp4096 |       228.27 ± 11.31 |

build: ace20f679 (11600)
