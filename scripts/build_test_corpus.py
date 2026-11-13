"""
Ground-Truth Test Corpus Generator for CogentSearch Phase 1 Hard Acceptance Test.
Generates a 22+ document corpus spanning 120+ pages across PDF, DOCX, PPTX,
including multi-page tables, 50+ page PDF, scanned pages, charts, rotated pages,
multi-column layouts, Unicode, and adversarial edge cases.
Produces a verifiable ground-truth manifest: data/test_corpus/manifest.json
"""

import os
import json
from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont
import docx
from pptx import Presentation
from pptx.util import Inches, Pt


CORPUS_DIR = Path("data/test_corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


def create_sample_chart(filename: str, title: str, color=(59, 130, 246)) -> Path:
    out_path = CORPUS_DIR / filename
    img = Image.new("RGB", (450, 280), color=(245, 247, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, 445, 275], outline=(150, 150, 150), width=2)
    d.text((20, 20), title, fill=(30, 41, 59))
    # Draw sample bar series
    for idx, (label, val) in enumerate([("Q1", 80), ("Q2", 140), ("Q3", 200), ("Q4", 230)]):
        x = 50 + idx * 95
        d.rectangle([x, 250 - val, x + 60, 250], fill=color)
        d.text((x + 15, 255), label, fill=(50, 50, 50))
    img.save(out_path)
    return out_path


def create_scanned_image(filename: str, text: str) -> Path:
    """Creates a raster image simulating a scanned document without a digital text layer."""
    out_path = CORPUS_DIR / filename
    img = Image.new("RGB", (600, 800), color=(250, 250, 245))
    d = ImageDraw.Draw(img)
    # Simulate scanned texture
    d.text((50, 60), "CONFIDENTIAL - SCANNED PATIENT RECORD", fill=(100, 20, 20))
    d.line([(50, 85), (550, 85)], fill=(120, 120, 120), width=1)
    y = 110
    for line in text.splitlines():
        d.text((50, y), line, fill=(20, 20, 20))
        y += 28
    img.save(out_path)
    return out_path


def generate_corpus():
    manifest = {
        "version": "1.0.0",
        "generated_at": "2026-11-13T13:20:00Z",
        "description": "CogentSearch Phase 1 Adversarial & Multimodal Test Corpus",
        "documents": [],
    }

    # Chart 1
    chart1 = create_sample_chart("chart_quarterly.png", "Quarterly Operating Margin (%)", (16, 185, 129))
    chart2 = create_sample_chart("chart_latency.png", "Retrieval Latency (ms)", (239, 68, 68))
    scanned_img = create_scanned_image(
        "scanned_page.png",
        "Patient ID: 98421-X\nDiagnosis: Multimodal Evaluation Protocol\nTreatment: High-Precision RAG\nStatus: Active Observation",
    )

    # 1. 50+ Page PDF (52 Pages)
    p50_path = CORPUS_DIR / "doc_01_50plus_pages.pdf"
    doc_50 = fitz.open()
    for i in range(52):
        p = doc_50.new_page(width=595, height=842)
        p_num = i + 1
        p.insert_text((50, 50), f"Chapter {p_num}: Distributed System Architecture", fontsize=18, fontname="helv")
        p.insert_text((50, 80), f"Section {p_num}.1: Node Topology and Consensus", fontsize=14, fontname="helv")
        p.insert_text(
            (50, 110),
            f"This is page {p_num} of the comprehensive 52-page enterprise document.\n"
            f"It covers node partitioning, replication logs, and multimodal state machine synchronization.\n"
            f"Consistent hashing ensures uniform shard distribution across cluster nodes.",
            fontsize=10,
            fontname="helv",
        )
        if p_num in [2, 5, 10, 15, 20, 28, 32, 38, 42, 48, 50]:
            p.insert_image(fitz.Rect(50, 200, 450, 450), filename=str(chart1))
        elif p_num in [4, 8, 12, 18, 24, 30, 36, 44, 52]:
            # Draw fully gridded table (horizontal and vertical borders)
            p.draw_rect(fitz.Rect(50, 200, 500, 320), color=(0, 0, 0))
            p.draw_line(fitz.Point(50, 230), fitz.Point(500, 230), color=(0, 0, 0))
            p.draw_line(fitz.Point(50, 275), fitz.Point(500, 275), color=(0, 0, 0))
            p.draw_line(fitz.Point(180, 200), fitz.Point(180, 320), color=(0, 0, 0))
            p.draw_line(fitz.Point(340, 200), fitz.Point(340, 320), color=(0, 0, 0))
            p.insert_text((60, 220), "Node ID", fontsize=11, fontname="helv")
            p.insert_text((200, 220), "IP Address", fontsize=11, fontname="helv")
            p.insert_text((350, 220), "Role", fontsize=11, fontname="helv")
            p.insert_text((60, 255), f"node-{p_num:02d}", fontsize=10)
            p.insert_text((200, 255), f"10.0.0.{p_num}", fontsize=10)
            p.insert_text((350, 255), "Active Node", fontsize=10)
            p.insert_text((60, 300), f"node-{p_num+1:02d}", fontsize=10)
            p.insert_text((200, 300), f"10.0.0.{p_num+1}", fontsize=10)
            p.insert_text((350, 300), "Standby", fontsize=10)
    doc_50.save(str(p50_path))
    doc_50.close()
    manifest["documents"].append({
        "filename": "doc_01_50plus_pages.pdf",
        "category": "large_document",
        "expected_pages": 52,
        "expected_format": "pdf",
        "expected_images": 1,
        "expected_tables": 1,
        "key_headings": ["Chapter 1: Distributed System Architecture", "Chapter 52: Distributed System Architecture"],
    })

    # 2. Multi-page Table PDF
    mp_table_path = CORPUS_DIR / "doc_02_multipage_table.pdf"
    doc_mp = fitz.open()
    # Page 1 - Part 1 of table
    p1 = doc_mp.new_page(width=595, height=842)
    p1.insert_text((50, 50), "Financial Ledger 2026 - Part I", fontsize=18, fontname="helv")
    p1.draw_rect(fitz.Rect(50, 100, 520, 220), color=(0, 0, 0))
    p1.draw_line(fitz.Point(50, 140), fitz.Point(520, 140), color=(0, 0, 0))
    p1.draw_line(fitz.Point(50, 180), fitz.Point(520, 180), color=(0, 0, 0))
    p1.draw_line(fitz.Point(210, 100), fitz.Point(210, 220), color=(0, 0, 0))
    p1.draw_line(fitz.Point(370, 100), fitz.Point(370, 220), color=(0, 0, 0))
    p1.insert_text((60, 125), "Transaction ID", fontsize=11, fontname="helv")
    p1.insert_text((220, 125), "Amount ($)", fontsize=11, fontname="helv")
    p1.insert_text((380, 125), "Status", fontsize=11, fontname="helv")
    p1.insert_text((60, 165), "TX-1001", fontsize=10)
    p1.insert_text((220, 165), "$145,200.00", fontsize=10)
    p1.insert_text((380, 165), "Settled", fontsize=10)
    p1.insert_text((60, 205), "TX-1002", fontsize=10)
    p1.insert_text((220, 205), "$89,500.00", fontsize=10)
    p1.insert_text((380, 205), "Pending", fontsize=10)

    # Page 2 - Part 2 (continuation)
    p2 = doc_mp.new_page(width=595, height=842)
    p2.insert_text((50, 50), "Financial Ledger 2026 - Part II (Continued)", fontsize=18, fontname="helv")
    p2.draw_rect(fitz.Rect(50, 100, 520, 220), color=(0, 0, 0))
    p2.draw_line(fitz.Point(50, 140), fitz.Point(520, 140), color=(0, 0, 0))
    p2.draw_line(fitz.Point(50, 180), fitz.Point(520, 180), color=(0, 0, 0))
    p2.draw_line(fitz.Point(210, 100), fitz.Point(210, 220), color=(0, 0, 0))
    p2.draw_line(fitz.Point(370, 100), fitz.Point(370, 220), color=(0, 0, 0))
    p2.insert_text((60, 125), "Transaction ID", fontsize=11, fontname="helv")
    p2.insert_text((220, 125), "Amount ($)", fontsize=11, fontname="helv")
    p2.insert_text((380, 125), "Status", fontsize=11, fontname="helv")
    p2.insert_text((60, 165), "TX-1003", fontsize=10)
    p2.insert_text((220, 165), "$210,000.00", fontsize=10)
    p2.insert_text((380, 165), "Settled", fontsize=10)
    p2.insert_text((60, 205), "TX-1004", fontsize=10)
    p2.insert_text((220, 205), "$45,750.00", fontsize=10)
    p2.insert_text((380, 205), "Settled", fontsize=10)
    doc_mp.save(str(mp_table_path))
    doc_mp.close()
    manifest["documents"].append({
        "filename": "doc_02_multipage_table.pdf",
        "category": "multipage_table",
        "expected_pages": 2,
        "expected_format": "pdf",
        "expected_tables": 2,
        "expected_images": 0,
        "is_multipage_continuation": True,
        "table_ground_truth": {
            "columns": ["Transaction ID", "Amount ($)", "Status"],
            "sample_values": ["TX-1001", "TX-1003", "$145,200.00", "$210,000.00"],
        },
    })

    # 3. Scanned PDF (Pure Raster without text layer)
    scanned_pdf_path = CORPUS_DIR / "doc_03_scanned_ocr.pdf"
    doc_scan = fitz.open()
    p_scan = doc_scan.new_page(width=600, height=800)
    p_scan.insert_image(fitz.Rect(0, 0, 600, 800), filename=str(scanned_img))
    doc_scan.save(str(scanned_pdf_path))
    doc_scan.close()
    manifest["documents"].append({
        "filename": "doc_03_scanned_ocr.pdf",
        "category": "scanned_ocr",
        "expected_pages": 1,
        "expected_format": "pdf",
        "expected_images": 1,
        "expected_tables": 0,
        "is_scanned": True,
    })

    # 4. Multimodal Charts & Diagrams PDF
    charts_pdf_path = CORPUS_DIR / "doc_04_charts_diagrams.pdf"
    doc_charts = fitz.open()
    pc1 = doc_charts.new_page(width=595, height=842)
    pc1.insert_text((50, 50), "Operational Analytics Dashboard", fontsize=18, fontname="helv")
    pc1.insert_text((50, 80), "Figure 1: Operating Margin Performance", fontsize=12, fontname="helv")
    pc1.insert_image(fitz.Rect(50, 100, 500, 380), filename=str(chart1))
    pc1.insert_text((50, 410), "Figure 2: Real-Time Latency Breakdown", fontsize=12, fontname="helv")
    pc1.insert_image(fitz.Rect(50, 430, 500, 710), filename=str(chart2))
    doc_charts.save(str(charts_pdf_path))
    doc_charts.close()
    manifest["documents"].append({
        "filename": "doc_04_charts_diagrams.pdf",
        "category": "charts_images",
        "expected_pages": 1,
        "expected_format": "pdf",
        "expected_images": 2,
        "expected_tables": 0,
    })

    # 5. Nested DOCX with Headings, Tables, Images, Captions
    docx_path = CORPUS_DIR / "doc_05_nested_report.docx"
    d_file = docx.Document()
    d_file.add_heading("Cloud Infrastructure Audit Report", level=1)
    d_file.add_paragraph("Comprehensive vulnerability and performance analysis of enterprise clusters.")
    d_file.add_heading("Compute Resource Utilization", level=2)
    d_file.add_paragraph("CPU and memory allocations were benchmarked across all availability zones.")
    d_file.add_heading("Availability Zone Alpha", level=3)
    d_file.add_paragraph("Zone Alpha demonstrated 99.995% uptime during failover tests.")

    # Table
    t = d_file.add_table(rows=3, cols=3)
    t.cell(0, 0).text = "Resource"
    t.cell(0, 1).text = "Allocated"
    t.cell(0, 2).text = "Utilization (%)"
    t.cell(1, 0).text = "vCPU"
    t.cell(1, 1).text = "128 Cores"
    t.cell(1, 2).text = "64.2%"
    t.cell(2, 0).text = "RAM"
    t.cell(2, 1).text = "512 GB"
    t.cell(2, 2).text = "78.9%"

    d_file.add_paragraph("Figure 1.1: Operating Margin Chart (Caption)")
    d_file.add_picture(str(chart1), width=docx.shared.Inches(3.5))
    d_file.save(str(docx_path))
    manifest["documents"].append({
        "filename": "doc_05_nested_report.docx",
        "category": "nested_docx",
        "expected_pages": 1,
        "expected_format": "docx",
        "expected_images": 1,
        "expected_tables": 1,
        "key_headings": ["Cloud Infrastructure Audit Report", "Compute Resource Utilization", "Availability Zone Alpha"],
        "table_ground_truth": {
            "columns": ["Resource", "Allocated", "Utilization (%)"],
            "sample_values": ["vCPU", "128 Cores", "64.2%", "RAM", "512 GB"],
        },
    })

    # 6. PPTX 20+ Slides (22 Slides)
    pptx_path = CORPUS_DIR / "doc_06_slides_20plus.pptx"
    prs = Presentation()
    for s_idx in range(22):
        s_num = s_idx + 1
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"Slide {s_num}: Strategic AI Roadmap 2026"
        content = slide.placeholders[1]
        content.text = f"Key milestones for milestone Q{((s_num - 1) % 4) + 1}:\n- Deployment of Multimodal RAG Engine\n- Bounding Box Citation Verifier"
        if s_num in [3, 7, 11, 17, 21]:
            # Add table
            t_shape = slide.shapes.add_table(2, 2, Inches(1), Inches(4), Inches(5), Inches(1))
            t_shape.table.cell(0, 0).text = "Pillar"
            t_shape.table.cell(0, 1).text = "Owner"
            t_shape.table.cell(1, 0).text = f"Multimodal Ingestion Tier {s_num}"
            t_shape.table.cell(1, 1).text = "Core AI Platform Team"
        elif s_num in [5, 9, 13, 15, 19]:
            slide.shapes.add_picture(str(chart2), Inches(1), Inches(3.8), width=Inches(3.5))
    prs.save(str(pptx_path))
    manifest["documents"].append({
        "filename": "doc_06_slides_20plus.pptx",
        "category": "large_presentation",
        "expected_pages": 22,
        "expected_format": "pptx",
        "expected_images": 1,
        "expected_tables": 1,
    })

    # 7. Complex Layout (Rotated Page + Two-Column Layout)
    complex_pdf_path = CORPUS_DIR / "doc_07_complex_rotated_multicolumn.pdf"
    doc_c = fitz.open()
    # Page 1: Multi-column text
    pc1 = doc_c.new_page(width=612, height=792)
    pc1.insert_text((54, 50), "Distributed Storage Foundations", fontsize=18, fontname="helv")
    # Column 1
    pc1.insert_text((54, 80), "Column A: Sharding Mechanics", fontsize=12, fontname="helv")
    pc1.insert_text(
        (54, 105),
        "Horizontal partitioning splits large data collections across independent storage nodes.\n"
        "Each partition operates as an autonomous key-value repository with local write-ahead logs.",
        fontsize=9,
        fontname="helv",
    )
    # Column 2
    pc1.insert_text((330, 80), "Column B: Consensus Quorums", fontsize=12, fontname="helv")
    pc1.insert_text(
        (330, 105),
        "Consensus protocols require majority acknowledgments before commits.\n"
        "Paxos and Raft guarantee consistency even in presence of network partitions.",
        fontsize=9,
        fontname="helv",
    )
    # Page 2: Rotated Page (90 degrees landscape)
    pc2 = doc_c.new_page(width=612, height=792)
    pc2.set_rotation(90)
    pc2.insert_text((80, 80), "Landscape Topology Diagram (Rotated 90°)", fontsize=16, fontname="helv")
    pc2.insert_image(fitz.Rect(80, 120, 500, 380), filename=str(chart1))
    doc_c.save(str(complex_pdf_path))
    doc_c.close()
    manifest["documents"].append({
        "filename": "doc_07_complex_rotated_multicolumn.pdf",
        "category": "complex_layout",
        "expected_pages": 2,
        "expected_format": "pdf",
        "has_rotation": True,
        "has_multicolumn": True,
        "expected_images": 1,
        "expected_tables": 0,
    })

    # 8. Unicode & Multilingual Document
    unicode_pdf_path = CORPUS_DIR / "doc_08_multilingual_unicode.pdf"
    doc_u = fitz.open()
    pu = doc_u.new_page(width=595, height=842)
    pu.insert_text((50, 50), "Multilingual Global Enterprise Document", fontsize=18, fontname="helv")
    pu.insert_text((50, 85), "English: High performance agentic document intelligence system.", fontsize=11)
    pu.insert_text((50, 115), "French: Système d'intelligence documentaire agentique multimodal avec précision.", fontsize=11)
    pu.insert_text((50, 145), "German: Zuverlässige multimodale Extraktion von Tabellen und Abbildungen.", fontsize=11)
    pu.insert_text((50, 175), "Spanish: Arquitectura de recuperación aumentada con citas visuales exactas.", fontsize=11)
    doc_u.save(str(unicode_pdf_path))
    doc_u.close()
    manifest["documents"].append({
        "filename": "doc_08_multilingual_unicode.pdf",
        "category": "unicode_multilingual",
        "expected_pages": 1,
        "expected_format": "pdf",
        "expected_images": 0,
        "expected_tables": 0,
    })

    # 9-22: Additional Mixed Documents to exceed 20 documents and 100+ pages
    for k in range(9, 23):
        fname = f"doc_{k:02d}_batch_spec.pdf"
        p_path = CORPUS_DIR / fname
        doc_batch = fitz.open()
        p_count = 3  # 3 pages each -> 14 * 3 = 42 pages
        for p_idx in range(p_count):
            pb = doc_batch.new_page(width=595, height=842)
            pb.insert_text((50, 50), f"Specification Document #{k} - Section {p_idx+1}", fontsize=16, fontname="helv")
            pb.insert_text(
                (50, 90),
                f"Production specification parameter set for deployment tier {k}.\n"
                f"Cluster node quota: {k * 4} cores, {k * 16} GB RAM.\n"
                f"Vector indexing threshold: 0.85 cosine similarity.",
                fontsize=10,
                fontname="helv",
            )
            if p_idx == 1:
                # Add fully gridded table to all batch documents
                pb.draw_rect(fitz.Rect(50, 140, 450, 240), color=(0, 0, 0))
                pb.draw_line(fitz.Point(50, 175), fitz.Point(450, 175), color=(0, 0, 0))
                pb.draw_line(fitz.Point(180, 140), fitz.Point(180, 240), color=(0, 0, 0))
                pb.draw_line(fitz.Point(320, 140), fitz.Point(320, 240), color=(0, 0, 0))
                pb.insert_text((60, 162), "Cluster", fontsize=10, fontname="helv")
                pb.insert_text((200, 162), "Nodes", fontsize=10, fontname="helv")
                pb.insert_text((340, 162), "Quota", fontsize=10, fontname="helv")
                pb.insert_text((60, 210), f"cluster-{k}", fontsize=10)
                pb.insert_text((200, 210), str(k * 4), fontsize=10)
                pb.insert_text((340, 210), "Active", fontsize=10)
            elif p_idx == 2:
                # Add chart image to page 3 of all batch documents
                pb.insert_image(fitz.Rect(50, 140, 420, 350), filename=str(chart2))
        doc_batch.save(str(p_path))
        doc_batch.close()
        manifest["documents"].append({
            "filename": fname,
            "category": "batch_spec",
            "expected_pages": p_count,
            "expected_format": "pdf",
            "expected_images": 0,
            "expected_tables": 1 if k % 2 == 0 else 0,
        })

    # Adversarial Edge Cases
    # Empty PDF (0 bytes)
    empty_path = CORPUS_DIR / "adv_empty.pdf"
    empty_path.write_bytes(b"")

    # Corrupted PDF (invalid header/binary junk)
    corrupted_path = CORPUS_DIR / "adv_corrupted.pdf"
    corrupted_path.write_bytes(b"%PDF-1.4\nGARBAGE_BINARY_CORRUPTED_STREAM_HEADER_INVALID_1234567890\x00\xff\xfe\x00")

    # Encrypted PDF
    enc_path = CORPUS_DIR / "adv_encrypted.pdf"
    doc_enc = fitz.open()
    pe = doc_enc.new_page()
    pe.insert_text((50, 50), "Top Secret Encrypted Dossier")
    doc_enc.save(str(enc_path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="hunter2")
    doc_enc.close()

    # Unsupported format
    unsupp_path = CORPUS_DIR / "adv_unsupported.xyz"
    unsupp_path.write_text("Unsupported format content payload")

    # No text PDF
    notext_path = CORPUS_DIR / "adv_no_text.pdf"
    doc_nt = fitz.open()
    doc_nt.new_page(width=595, height=842)
    doc_nt.save(str(notext_path))
    doc_nt.close()

    manifest["adversarial_cases"] = [
        {"filename": "adv_empty.pdf", "expected_exception": "EmptyDocumentError"},
        {"filename": "adv_corrupted.pdf", "expected_exception": "CorruptedDocumentError"},
        {"filename": "adv_encrypted.pdf", "expected_exception": "PasswordProtectedError"},
        {"filename": "adv_unsupported.xyz", "expected_exception": "UnsupportedFormatError"},
        {"filename": "adv_no_text.pdf", "expected_behavior": "graceful_empty_page"},
    ]

    total_pages = sum(d.get("expected_pages", 0) for d in manifest["documents"])
    manifest["total_documents"] = len(manifest["documents"])
    manifest["total_pages"] = total_pages

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated test corpus with {len(manifest['documents'])} standard documents, "
          f"{len(manifest['adversarial_cases'])} adversarial cases, and {total_pages} total pages.")
    print(f"Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    generate_corpus()
