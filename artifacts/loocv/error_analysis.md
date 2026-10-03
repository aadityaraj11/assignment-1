# Error Analysis — REAL GEMINI LOOCV

Detailed breakdown of incorrect predictions across Leave-One-Out Cross-Validation.

## Document: `6683127-House-Rental-Contract-GERALDINE-GALINATO-v2-Page-1`
**Top Retrieved Examples**:
- `6683129-House-Rental-Contract-Geraldine-Galinato-v2` (similarity: 0.7905)
- `50070534-RENTAL-AGREEMENT` (similarity: 0.5916)
- `47854715-RENTAL-AGREEMENT` (similarity: 0.5568)

**Failed Fields**:
- **Field**: `party_two`
  - **Expected**: `GERALDINE Q. GALINATO`
  - **Predicted**: `GERALDINE O. GALINATO`
  - **Failure Category**: `party identification ambiguity`

## Document: `18325926-Rental-Agreement-1`
**Top Retrieved Examples**:
- `50070534-RENTAL-AGREEMENT` (similarity: 0.8220)
- `54945838-Rental-Agreement` (similarity: 0.7989)
- `47854715-RENTAL-AGREEMENT` (similarity: 0.7710)

**Failed Fields**:
- **Field**: `agreement_value`
  - **Expected**: `4000`
  - **Predicted**: `4000.00`
  - **Failure Category**: `multiple monetary values`

- **Field**: `agreement_end_date`
  - **Expected**: `31.11.2009`
  - **Predicted**: `04.11.2009`
  - **Failure Category**: `multiple dates`

- **Field**: `party_two`
  - **Expected**: `P.M. Narayana Namboodri`
  - **Predicted**: `Sri. P.M. Narayana Namboodri`
  - **Failure Category**: `party identification ambiguity`

## Document: `36199312-Rental-Agreement`
**Top Retrieved Examples**:
- `54770958-Rental-Agreement` (similarity: 0.7031)
- `54945838-Rental-Agreement` (similarity: 0.6873)
- `44737744-Maddireddy-Bhargava-Reddy-Rental-Agreement` (similarity: 0.6703)

**Failed Fields**:
- **Field**: `agreement_value`
  - **Expected**: `3800`
  - **Predicted**: `38000`
  - **Failure Category**: `multiple monetary values`

- **Field**: `agreement_end_date`
  - **Expected**: `31.04.2011`
  - **Predicted**: `31.03.2011`
  - **Failure Category**: `multiple dates`

- **Field**: `party_one`
  - **Expected**: `Balaji.R`
  - **Predicted**: `Mr. Balaji.R`
  - **Failure Category**: `party identification ambiguity`

- **Field**: `party_two`
  - **Expected**: `Kartheek R`
  - **Predicted**: `Mr. Kartheek R`
  - **Failure Category**: `party identification ambiguity`

## Document: `44737744-Maddireddy-Bhargava-Reddy-Rental-Agreement`
**Top Retrieved Examples**:
- `18325926-Rental-Agreement-1` (similarity: 0.6729)
- `36199312-Rental-Agreement` (similarity: 0.6703)
- `50070534-RENTAL-AGREEMENT` (similarity: 0.6585)

**Failed Fields**:
- **Field**: `agreement_value`
  - **Expected**: `3000`
  - **Predicted**: `999.79`
  - **Failure Category**: `multiple monetary values`

- **Field**: `agreement_start_date`
  - **Expected**: `20.09.2010`
  - **Predicted**: `None`
  - **Failure Category**: `multiple dates`

- **Field**: `agreement_end_date`
  - **Expected**: `19.07.2011`
  - **Predicted**: `None`
  - **Failure Category**: `multiple dates`

- **Field**: `party_one`
  - **Expected**: `M.V.V. VIJAYA SHANKAR`
  - **Predicted**: `M. V, V« VI JAY A SHANKAR`
  - **Failure Category**: `party identification ambiguity`

## Document: `47854715-RENTAL-AGREEMENT`
**Top Retrieved Examples**:
- `54945838-Rental-Agreement` (similarity: 0.8980)
- `54770958-Rental-Agreement` (similarity: 0.8472)
- `18325926-Rental-Agreement-1` (similarity: 0.7710)

**Failed Fields**:
- **Field**: `agreement_end_date`
  - **Expected**: `31.02.2011`
  - **Predicted**: `28.02.2011`
  - **Failure Category**: `multiple dates`

- **Field**: `renewal_notice_days`
  - **Expected**: `60`
  - **Predicted**: `30`
  - **Failure Category**: `model extraction error`

- **Field**: `party_one`
  - **Expected**: `P C MATHEW`
  - **Predicted**: `P C Mathew`
  - **Failure Category**: `party identification ambiguity`

- **Field**: `party_two`
  - **Expected**: `L GOPINATH`
  - **Predicted**: `L Gopinath`
  - **Failure Category**: `party identification ambiguity`

## Document: `54770958-Rental-Agreement`
**Top Retrieved Examples**:
- `54945838-Rental-Agreement` (similarity: 0.8479)
- `47854715-RENTAL-AGREEMENT` (similarity: 0.8472)
- `18325926-Rental-Agreement-1` (similarity: 0.7383)

**Failed Fields**:
- **Field**: `agreement_value`
  - **Expected**: `8000`
  - **Predicted**: `5500`
  - **Failure Category**: `multiple monetary values`

- **Field**: `agreement_start_date`
  - **Expected**: `01.04.2011`
  - **Predicted**: `20.04.2011`
  - **Failure Category**: `multiple dates`

- **Field**: `agreement_end_date`
  - **Expected**: `31.03.2012`
  - **Predicted**: `19.03.2012`
  - **Failure Category**: `multiple dates`

- **Field**: `renewal_notice_days`
  - **Expected**: `90`
  - **Predicted**: `60`
  - **Failure Category**: `model extraction error`

- **Field**: `party_one`
  - **Expected**: `K. Parthasarathy`
  - **Predicted**: `Mrs.Asha Ramesh & Mr.Ramesh. K.N.`
  - **Failure Category**: `party identification ambiguity`

- **Field**: `party_two`
  - **Expected**: `Veerabrahmam Bathini`
  - **Predicted**: `Mrs.Sadasivuni Deepti & Mr.Sadasivuni Kiran`
  - **Failure Category**: `party identification ambiguity`

## Document: `54945838-Rental-Agreement`
**Top Retrieved Examples**:
- `47854715-RENTAL-AGREEMENT` (similarity: 0.8980)
- `54770958-Rental-Agreement` (similarity: 0.8479)
- `18325926-Rental-Agreement-1` (similarity: 0.7989)

**Failed Fields**:
- **Field**: `agreement_value`
  - **Expected**: `5500`
  - **Predicted**: `8000`
  - **Failure Category**: `multiple monetary values`

- **Field**: `agreement_start_date`
  - **Expected**: `21.04.2011`
  - **Predicted**: `01.04.2011`
  - **Failure Category**: `execution date vs commencement date`

- **Field**: `agreement_end_date`
  - **Expected**: `19.02.2012`
  - **Predicted**: `31.03.2012`
  - **Failure Category**: `execution date vs commencement date`

- **Field**: `renewal_notice_days`
  - **Expected**: `60`
  - **Predicted**: `90`
  - **Failure Category**: `model extraction error`

- **Field**: `party_one`
  - **Expected**: `Asha Ramesh & Ramesh K.N`
  - **Predicted**: `K. Parthasarathy`
  - **Failure Category**: `party identification ambiguity`

- **Field**: `party_two`
  - **Expected**: `Sadasivuni Deepthi & Sadasivuni Kiran`
  - **Predicted**: `Veerabrahmam Bathini`
  - **Failure Category**: `party identification ambiguity`

