# Publication review

## Included

MS MARCO search application, English project documentation, benchmark scripts and measured results, and a self-contained HTML dashboard. Query text and passage IDs in benchmark results originate from the public MS MARCO dev set; the repository does not include the passage corpus or embeddings.

## Excluded

Credentials and configuration files, environment files, personal absolute paths, account identifiers and billing records, raw conversation and internal working notes, downloaded datasets and vectors, screenshots not visually reviewed for publication, and the separate ADHD application and literature research workspace.

## Validation and limits

Reviewed the release text for common credential formats, email addresses, personal paths, credential files, and financial account records. No matched credential values were found in the release. The dashboard preserves measured retrieval results and caveats. The application remains NumPy 512d float32; MLX is a benchmark recommendation. A pattern scan cannot prove the absence of every form of sensitive information. No new retrieval benchmark or full test-suite run was performed for publication.
