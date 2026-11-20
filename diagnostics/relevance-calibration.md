# Initial relevance cutoff check

Cutoff: cosine similarity >= 0.60. Selected using these eight development examples; not an independent evaluation.

| Query | Expected behavior | Highest cosine |
| --- | --- | ---: |
| What was the Manhattan Project? | Return passages | 0.7457 |
| Who was Antonin Dvorak? | Return passages | 0.8044 |
| What is lactose made of? | Return passages | 0.6643 |
| How do you roast squash seeds? | Return passages | 0.7898 |
| How to deal with burnout | Return no passages | 0.3716 |
| What is the weather in Boston right now? | Return no passages | 0.5442 |
| What is the plot of the fictional novel The Glass Orchard of Zynthera? | Return no passages | 0.4040 |
| How do I install the fictional Flarnix 9000 quantum toaster? | Return no passages | 0.3394 |

The threshold separated these examples but is not a universal relevance detector. Test on a larger labeled set before claiming accuracy.
