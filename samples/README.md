# samples/

Bundled single-sample VCF files for the demo site. The numbered samples appear under "Don't have a file? Try a sample" and show their result instantly; upload `sample_snp_upload.vcf` to try the live file-upload path.

| File | Target population | Expected confidence | Shown in UI |
|---|---|---|---|
| [sample_snp_1.vcf](sample_snp_1.vcf) | YRI — Yoruba in Ibadan, Nigeria | ~93% | Yes |
| [sample_snp_2.vcf](sample_snp_2.vcf) | CHB — Han Chinese in Beijing | ~91% | Yes |
| [sample_snp_3.vcf](sample_snp_3.vcf) | CEU — Northern Europeans from Utah | ~91% | Yes |
| [sample_snp_upload.vcf](sample_snp_upload.vcf) | JPT — Japanese in Tokyo | ~91% | No (upload demo only) |

## Format notes

- VCF 4.2, GRCh37 reference, chromosome 15
- ~1100–1150 bi-allelic SNPs per file, all phased (`0|0`, `0|1`, `1|1`)
- Variants are a subset of the model's feature registry; remaining registry features are imputed from training-set medians
- Confidence values are model predictions — minor variation is expected across deployments

## Precomputed results

Each numbered sample has a `sample_snp_N.result.json` beside it: the exact `POST /predict` response from the deployed model, which the site renders without calling the API. Regenerate them with `bin/06_precompute_samples.sh` after every deploy that binds a new `ModelRunId`, and commit them. All result files must carry the same `model_artifact_version` (a test checks this), and the site build fails if a numbered sample has no result file.
