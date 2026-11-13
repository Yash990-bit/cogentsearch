# Citation Integrity Audit Report
## Phase 1 Hard Acceptance Criterion: 50-Element Provenance Audit

- **Audit Sample Size**: 50 randomly sampled visual elements (Seed: 42)
- **Audit Result**: PASS
- **Pass Rate**: 100.0% (50/50 passed)
- **Mismatches Detected**: 0

---

### Audit Checklist Criteria Evaluated per Element:
1. `source_file_exists`: Validated that source document exists in test corpus.
2. `page_in_range`: Validated page/slide number is within valid document bounds [1, total_pages].
3. `id_traceability`: Validated element ID encodes document ID and page index deterministically.
4. `modality_valid`: Validated visual element classification is valid enum (image/table/chart).
5. `bounding_box_valid`: Validated 0 <= x0 < x1 and 0 <= y0 < y1 within document dimensions.
6. `crop_file_valid`: Validated cropped image file exists on disk with non-zero byte size and valid image header.
7. `linked_to_retrieval_chunk`: Validated element ID is linked to at least one injected retrieval chunk.

---

### Sample of Audited Elements (Top 20 of 50):

| # | Element ID | Source Document | Page | Modality | Bounding Box [x0, y0, x1, y1] | Crop Valid | Chunk Linked | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `img_fff35c0eb4f1_p38_0` | `doc_01_50plus_pages.pdf` | 38 | image | [50, 201, 450, 449] | ✔ | ✔ | **PASS** |
| 2 | `img_1ef2a446fb84_p3_0` | `doc_18_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 3 | `img_587f7100a417_p3_0` | `doc_14_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 4 | `img_78f40b1eca73_p3_0` | `doc_16_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 5 | `img_f7a6c59dd98c_s15_0` | `doc_06_slides_20plus.pptx` | 15 | image | [72, 274, 324, 430] | ✔ | ✔ | **PASS** |
| 6 | `table_f7a6c59dd98c_s11_0` | `doc_06_slides_20plus.pptx` | 11 | table | [72, 288, 432, 360] | ✔ | ✔ | **PASS** |
| 7 | `img_f7a6c59dd98c_s9_0` | `doc_06_slides_20plus.pptx` | 9 | image | [72, 274, 324, 430] | ✔ | ✔ | **PASS** |
| 8 | `img_6e43a3e445f8_p1_0` | `sample_enterprise_report.pdf` | 1 | image | [69, 200, 485, 450] | ✔ | ✔ | **PASS** |
| 9 | `img_c55964846162_p3_0` | `doc_20_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 10 | `img_45413dbe1a76_p3_0` | `doc_10_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 11 | `img_fff35c0eb4f1_p48_0` | `doc_01_50plus_pages.pdf` | 48 | image | [50, 201, 450, 449] | ✔ | ✔ | **PASS** |
| 12 | `img_3fc6962b26bf_p3_0` | `doc_12_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 13 | `img_fff35c0eb4f1_p20_0` | `doc_01_50plus_pages.pdf` | 20 | image | [50, 201, 450, 449] | ✔ | ✔ | **PASS** |
| 14 | `img_332c5c813418_p2_0` | `doc_07_complex_rotated_multicolumn.pdf` | 2 | image | [81, 120, 499, 380] | ✔ | ✔ | **PASS** |
| 15 | `table_fff35c0eb4f1_p30_0` | `doc_01_50plus_pages.pdf` | 30 | table | [60, 212, 404, 302] | ✔ | ✔ | **PASS** |
| 16 | `table_fff35c0eb4f1_p4_0` | `doc_01_50plus_pages.pdf` | 4 | table | [60, 212, 404, 302] | ✔ | ✔ | **PASS** |
| 17 | `img_2c5af7646eae_p3_0` | `doc_11_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 18 | `img_c26b3d7fbbff_p3_0` | `doc_17_batch_spec.pdf` | 3 | image | [66, 140, 404, 350] | ✔ | ✔ | **PASS** |
| 19 | `table_fff35c0eb4f1_p52_0` | `doc_01_50plus_pages.pdf` | 52 | table | [60, 212, 404, 302] | ✔ | ✔ | **PASS** |
| 20 | `table_f7a6c59dd98c_s7_0` | `doc_06_slides_20plus.pptx` | 7 | table | [72, 288, 432, 360] | ✔ | ✔ | **PASS** |

> [!NOTE]
> **Zero Mismatches Detected**: 100% of the 50 audited visual elements demonstrated exact coordinate integrity, valid crop assets, and complete retrieval chunk provenance linking.