# Version 6 reporting update

All four managerial studies now use `pooled_orders`: average model treatment effects across option orders, calculate absolute differences from human effects, divide by the response scale span, and average equally across effects. This replaces the mixed reporting conventions in v5.

| Study | Prompt normalized error | SAE normalized error | Relative reduction |
| --- | ---: | ---: | ---: |
| Director compensation | 13.22% | 4.52% | 65.83% |
| Employee voice | 13.43% | 7.33% | 45.43% |
| CEO trust source | 2.67% | 1.40% | 47.42% |
| Restaurant trust transfer | 5.21% | 3.94% | 24.31% |

This change affects reporting only. Saved observations, intervention assets, human targets, and original controller fitting and selection are unchanged. Per-order diagnostics remain available in the output records. Previous v5 archives retain their original calculations for provenance.
