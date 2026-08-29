# PaperBench suites

Manual grouping of PaperBench's 20 papers (`experiments/splits/all.txt` in the frontier-evals clone)
into 4-paper suites. No embedding clustering: at N=20, a domain label read off each paper is more
trustworthy than a clustering of 20 points.

| id | suite |
|----|-------|
| `0` | Reinforcement learning |
| `1` | Vision, foundation-model adaptation |
| `2` | NLP and large language models |
| `4` | Generative and probabilistic inference, diffusion and score-based |
| `5` | Diverse residual |

Suites 0, 1, 2 and 4 are the coherent ones and the main average pools them, 16 papers. Suite 5 is a
low-reuse probe: variational inference, a PINN loss landscape, coreset selection and a black-box LLM
adapter share almost nothing. It asks whether SLA extracts something spurious when there is nothing
to share, so it is reported on its own rather than pooled.

`bbox` is in suite 5 and not 2 because BBox-Adapter only has API access to its model. The other four
LLM papers load a Hugging Face checkpoint and touch weights or hidden states, which is the pattern
that gets extracted; `bbox` shares none of it.

`id 3` is suites 0, 1 and 2 together in one 12-paper run. It is not a sixth suite, and pooling it
with the others counts each of those papers twice.
