# Projector Weights

This directory is reserved for SVD/GEVD projector files used at evaluation time.

Expected layout:

```text
projectors/svd/no_mean_sub_uu_positive.pt
projectors/svd/no_mean_sub_uu_negative.pt
projectors/gevd/no_mean_sub_uu_positive.pt
projectors/gevd/no_mean_sub_uu_negative.pt
```

These files are large and should be distributed through Git LFS, GitHub Releases, or an anonymous external artifact link during review.
