# Outlier Analysis Report

Outliers are detected using the IQR rule.

| feature                |     q1 |      q3 |     iqr |   lower_bound |   upper_bound |   outlier_count |   outlier_rate |
|:-----------------------|-------:|--------:|--------:|--------------:|--------------:|----------------:|---------------:|
| num_dependents         |    1   |    1    |    0    |          1    |          1    |             155 |          0.155 |
| credit_amount          | 1365.5 | 3972.25 | 2606.75 |      -2544.62 |       7882.38 |              72 |          0.072 |
| duration               |   12   |   24    |   12    |         -6    |         42    |              70 |          0.07  |
| age                    |   27   |   42    |   15    |          4.5  |         64.5  |              23 |          0.023 |
| existing_credits       |    1   |    2    |    1    |         -0.5  |          3.5  |               6 |          0.006 |
| installment_commitment |    2   |    4    |    2    |         -1    |          7    |               0 |          0     |
| residence_since        |    2   |    4    |    2    |         -1    |          7    |               0 |          0     |
