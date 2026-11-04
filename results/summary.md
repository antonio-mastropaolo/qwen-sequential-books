# Summary: Qwen/Qwen2.5-7B

Base: A=0.003, B=0.004. After stage 1 (book A): A=0.942, B=0.054.

| method | seeds | A after A | A after B | B after B | forgetting of A |
|---|---|---|---|---|---|
| naive | 3 | 0.942 ± 0.036 | 0.300 ± 0.008 | 0.977 ± 0.017 | 0.642 ± 0.034 |
| kl_global | 3 | 0.942 ± 0.036 | 0.820 ± 0.017 | 0.442 ± 0.024 | 0.122 ± 0.021 |
| replay_kl | 3 | 0.942 ± 0.036 | 0.957 ± 0.021 | 0.886 ± 0.063 | -0.015 ± 0.017 |

forgetting of A = acc_A(after A) - acc_A(after B). Values are mean ± sd over seeds.
