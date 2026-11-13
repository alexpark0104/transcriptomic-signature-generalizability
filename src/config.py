# Inherited candidate signature (NOT derived by this repository's pipeline).
#
# Provenance: undocumented. The signature was present in the repository's
# initial commit with no publication, script, or analysis explaining where it
# came from. Its derivation therefore cannot be shown to be independent of the
# external cohort, so its external result is reported separately and is not
# treated as an unbiased validation.
#
# What has been verified (src/check_inherited_signature.py):
#   * The Entrez IDs below are the genes these coefficients were fitted on.
#     Scoring the discovery patients with these IDs and coefficients gives an
#     in-sample AUC of 0.980, above all of 2,000 random 10-gene sets and all
#     of 2,000 coefficient shuffles (p = 0.0005 each).
#   * The gene symbols formerly stored here were wrong for 7 of the 10 genes
#     (checked against NCBI Gene) and have been removed. Symbols now come only
#     from data/entrez_to_symbol.csv, which records its source and date.
#
# The coefficients apply to z-scored expression.
SIGNATURE_GENES = [401613, 6017, 90737, 260436, 8710, 10991, 255189, 1118, 388533, 3358]
COEFFICIENTS = [0.539, -0.596, 0.641, -0.005, 0.033, -0.036, -0.791, -0.021, -0.332, -0.029]