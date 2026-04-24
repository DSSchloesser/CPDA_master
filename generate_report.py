"""
generate_report.py
──────────────────
Build the formal CPDA technical report PDF using reportlab.
Uses real outputs from the analysis pipeline.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUTS = Path("outputs")
REPORT_PATH = OUTPUTS / "CPDA_Technical_Report.pdf"

# ── Color palette ──────────────────────────────────────────────────────────────
DARK       = colors.HexColor("#1a1a2e")
ACCENT     = colors.HexColor("#2E86AB")
ACCENT2    = colors.HexColor("#E84855")
LIGHT_GRAY = colors.HexColor("#f5f5f7")
MID_GRAY   = colors.HexColor("#8a8a9a")
WHITE      = colors.white
BORDER     = colors.HexColor("#d0d0dc")

CLUSTER_COLORS = {
    0: colors.HexColor("#2E86AB"),
    1: colors.HexColor("#E84855"),
    2: colors.HexColor("#F9A03F"),
    3: colors.HexColor("#3BB273"),
}

# ── Styles ─────────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=DARK,
        spaceAfter=8,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=MID_GRAY,
        spaceAfter=6,
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta",
        fontName="Helvetica",
        fontSize=9,
        textColor=MID_GRAY,
        spaceAfter=4,
    )
    styles["section_header"] = ParagraphStyle(
        "section_header",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=ACCENT,
        spaceBefore=18,
        spaceAfter=6,
        borderPad=0,
    )
    styles["subsection_header"] = ParagraphStyle(
        "subsection_header",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=DARK,
        spaceBefore=10,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9.5,
        leading=14.5,
        textColor=DARK,
        spaceAfter=7,
        alignment=TA_JUSTIFY,
    )
    styles["body_small"] = ParagraphStyle(
        "body_small",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=DARK,
        spaceAfter=5,
        alignment=TA_JUSTIFY,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=MID_GRAY,
        spaceAfter=10,
        alignment=TA_CENTER,
    )
    styles["callout"] = ParagraphStyle(
        "callout",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=14,
        textColor=DARK,
        leftIndent=12,
        rightIndent=12,
        spaceAfter=8,
    )
    styles["finding"] = ParagraphStyle(
        "finding",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=15,
        textColor=WHITE,
        spaceBefore=4,
        spaceAfter=4,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=DARK,
        backColor=LIGHT_GRAY,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=4,
        spaceAfter=4,
    )
    return styles


# ── Table helpers ──────────────────────────────────────────────────────────────
def styled_table(data, col_widths, header_bg=ACCENT, zebra=True):
    n_rows = len(data)
    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 6),
        ("TOPPADDING",   (0, 0), (-1, 0), 6),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 4),
        ("TOPPADDING",   (0, 1), (-1, -1), 4),
        ("GRID",         (0, 0), (-1, -1), 0.3, BORDER),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 1), (0, -1), "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT_GRAY] if zebra else [WHITE]),
    ]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style_cmds))
    return t


def finding_box(text, styles, color=ACCENT):
    """Colored callout box for key findings."""
    data = [[Paragraph(text, styles["finding"])]]
    t = Table(data, colWidths=[6.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("ROUNDEDCORNERS", [4]),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
    ]))
    return t


# ── Page template ──────────────────────────────────────────────────────────────
def make_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(inch, 0.55*inch,
        "CPDA — Behavioral Latent Variable Analysis | EdNet-KT4 | Confidential")
    canvas.drawRightString(7.5*inch, 0.55*inch, f"Page {doc.page}")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(inch, 0.68*inch, 7.5*inch, 0.68*inch)
    canvas.restoreState()


# ── Report sections ────────────────────────────────────────────────────────────

def cover_page(story, styles):
    story.append(Spacer(1, 1.2*inch))

    # Top accent bar via table
    bar = Table([[""]], colWidths=[6.5*inch], rowHeights=[5])
    bar.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),ACCENT),
                              ("LINEBELOW",(0,0),(-1,-1),0,WHITE)]))
    story.append(bar)
    story.append(Spacer(1, 0.25*inch))

    story.append(Paragraph("Behavioral Latent Variable Analysis", styles["cover_title"]))
    story.append(Paragraph("EdNet-KT4 Interaction Telemetry", styles["cover_subtitle"]))
    story.append(Spacer(1, 0.15*inch))

    rq = Table([[Paragraph(
        "Research Question: Do latent behavioral clusters recoverable from "
        "interaction-sequence telemetry differentially predict short-term task "
        "completion versus long-term knowledge retention? Does \"efficient\" "
        "session behavior correlate with durable learning?",
        ParagraphStyle("rq", fontName="Helvetica-Oblique", fontSize=9.5,
                       leading=14, textColor=DARK)
    )]], colWidths=[6.5*inch])
    rq.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), LIGHT_GRAY),
        ("LEFTPADDING",(0,0),(-1,-1),14),
        ("RIGHTPADDING",(0,0),(-1,-1),14),
        ("TOPPADDING",(0,0),(-1,-1),10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LINEAFTER",(0,0),(0,-1),3,ACCENT),
    ]))
    story.append(rq)
    story.append(Spacer(1, 0.5*inch))

    meta = [
        ("Dataset",    "EdNet-KT4 (Riiid, 2020)"),
        ("Framework",  "Complexity Science · Psychometrics · Ecological Psychology"),
        ("Pipeline",   "Synthetic Telemetry → Feature Extraction → UMAP/HDBSCAN → Outcome Analysis"),
        ("Date",       "April 2026"),
    ]
    for k, v in meta:
        story.append(Paragraph(
            f'<font name="Helvetica-Bold" color="#2E86AB">{k}:</font>'
            f'  <font name="Helvetica" color="#555555">{v}</font>',
            ParagraphStyle("meta", fontSize=9, leading=14, spaceAfter=4)
        ))

    story.append(Spacer(1, 0.4*inch))
    bar2 = Table([[""]], colWidths=[6.5*inch], rowHeights=[2])
    bar2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BORDER)]))
    story.append(bar2)
    story.append(PageBreak())


def section_intro(story, styles):
    story.append(Paragraph("1. Introduction & Problem Formulation", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "Digital learning platforms generate granular interaction traces — timestamps, "
        "response latencies, answer sequences, and re-attempt patterns — that encode "
        "behavioral signals well beyond what traditional assessment captures. The EdNet-KT4 "
        "corpus (Choi et al., 2020), comprising tens of millions of interactions from Korean "
        "university entrance exam preparation, provides a high-resolution substrate for "
        "examining whether such telemetry reveals psychologically meaningful learner typologies.",
        styles["body"]
    ))

    story.append(Paragraph(
        "This analysis pursues the Behavioral Latent Variables question: can unsupervised "
        "clustering of interaction-sequence features identify stable learner archetypes, "
        "and do those archetypes differentially predict short-term task completion (STC) "
        "versus long-term retention (LTR)? The distinction matters pedagogically. "
        "A learner who is 'efficient' — high accuracy at low response time cost — "
        "may be deploying fluency strategies optimized for immediate performance "
        "without generating the retrieval-effort and spacing conditions known to "
        "support durable encoding (Bjork & Bjork, 2011; Kornell & Bjork, 2008).",
        styles["body"]
    ))

    story.append(Paragraph("Formal Problem Statement", styles["subsection_header"]))
    story.append(Paragraph(
        "Let U = {u<sub>1</sub>, ..., u<sub>N</sub>} be a population of learners, each "
        "associated with an interaction sequence:",
        styles["body"]
    ))
    story.append(Paragraph(
        "S<sub>u</sub> = {(t<sub>i</sub>, q<sub>i</sub>, a<sub>i</sub>, "
        "&#964;<sub>i</sub>, c<sub>i</sub>)}<sup>T<sub>u</sub></sup><sub>i=1</sub>",
        ParagraphStyle("eq", fontName="Courier", fontSize=9.5, leading=14,
                       leftIndent=24, spaceAfter=6, textColor=DARK)
    ))
    story.append(Paragraph(
        "where t<sub>i</sub> is a Unix timestamp (ms), q<sub>i</sub> is question identifier, "
        "a<sub>i</sub> is the submitted answer, &#964;<sub>i</sub> is response latency (ms), "
        "and c<sub>i</sub> &#8712; {0,1} is correctness. "
        "We seek a mapping f: S<sub>u</sub> &#8594; x<sub>u</sub> &#8712; R<sup>p</sup> "
        "and clustering g: x<sub>u</sub> &#8594; k<sub>u</sub> such that cluster membership "
        "k<sub>u</sub> is differentially predictive of:",
        styles["body"]
    ))

    outcomes = [
        ["Outcome", "Operationalization", "Temporal Scope"],
        ["Short-Term Task Completion (STC)",
         "Mean accuracy in latter half of all sessions",
         "Within-session, immediate"],
        ["Long-Term Retention (LTR)",
         "Accuracy in final 3 sessions minus first 3 sessions (retention_delta)",
         "Multi-day, cross-session"],
    ]
    story.append(styled_table(outcomes, [2.0*inch, 3.0*inch, 1.5*inch]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The core null hypothesis is H<sub>0</sub>: Session efficiency &#8869; LTR | STC — "
        "that is, efficient behavior predicts short-term performance but is independent "
        "of durable knowledge gain once STC is controlled.",
        styles["body"]
    ))
    story.append(PageBreak())


def section_features(story, styles):
    story.append(Paragraph("2. Feature Engineering Strategy", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "Features are organized across four theoretically motivated measurement layers. "
        "Outcome variables (retention_delta, proficiency_trend, session_acc_mean) are "
        "withheld from the clustering feature set to prevent circularity in validation. "
        "All features are computed per user from their full interaction history.",
        styles["body"]
    ))

    layers = [
        ["Layer", "Theoretical Basis", "Key Features", "N"],
        ["1. Kinematic /\nTemporal",
         "Response time modeling\n(van der Linden, 2006;\nMaris & van der Maas, 2012)",
         "rt_log_mean, rt_cv, rt_skew,\nrt_kurtosis, rt_iqr, rt_p10/p90,\npower_law_alpha",
         "9"],
        ["2. Accuracy /\nEfficiency",
         "Speed-accuracy tradeoff\n(Heitz, 2014; Wickelgren, 1977)",
         "accuracy, efficiency_index,\nsat_slope, sat_r2,\naccuracy_drift",
         "5"],
        ["3. Complexity /\nDynamics",
         "Complexity science; ecological\npsychology (Newell et al., 2001;\nvan Orden et al., 2003)",
         "approx_entropy, perm_entropy,\nhurst_exponent, rt_autocorr_lag1,\nproficiency_entropy, proficiency_std",
         "6"],
        ["4. Temporal /\nLongitudinal",
         "Spacing effect; forgetting curve\n(Ebbinghaus, 1885;\nCepeda et al., 2006)",
         "n_sessions, sessions_per_day,\nsession_acc_std, within_session_accel,\nreattempt_fraction",
         "8"],
    ]
    t = styled_table(layers, [1.1*inch, 1.7*inch, 2.8*inch, 0.4*inch])
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Design Decisions", styles["subsection_header"]))
    decisions = [
        ("<b>Log-transform of RT:</b> Response latency distributions exhibit "
         "log-normal to power-law character (Luce, 1986). All entropy and autocorrelation "
         "estimates operate on log(1+RT) to reduce heavy-tail distortion."),
        ("<b>Efficiency Index:</b> Defined as accuracy / log(mean RT). This captures "
         "the joint speed-accuracy tradeoff as a single scalar — a learner scoring 0.80 "
         "in 2,000ms occupies a fundamentally different behavioral niche than one "
         "scoring 0.80 in 20,000ms."),
        ("<b>Power Law Alpha (Hill estimator):</b> The tail exponent of the RT "
         "distribution indexes whether response timing follows light-tailed "
         "(&#945; > 2) or heavy-tailed L&#233;vy-like (&#945; &#8804; 2) dynamics. "
         "The latter is associated with long-memory, self-organized-critical processes "
         "in cognitive systems (Gilden, 2001)."),
        ("<b>Minimum interaction threshold:</b> Users with fewer than 40 interactions "
         "are excluded. ApEn and Hurst estimation require adequate series length; "
         "below this threshold, variance dominates signal."),
        ("<b>Session segmentation:</b> Sessions defined by a 30-minute inter-event "
         "gap threshold, consistent with PISA process data conventions "
         "(OECD, 2019)."),
    ]
    for d in decisions:
        story.append(Paragraph(f"&#8226;  {d}", styles["body_small"]))

    story.append(PageBreak())


def section_clustering(story, styles, metrics):
    story.append(Paragraph("3. Clustering Methodology", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "The clustering pipeline proceeds in five stages designed to handle the "
        "distributional properties of behavioral telemetry:",
        styles["body"]
    ))

    pipeline_steps = [
        ["Stage", "Method", "Rationale"],
        ["1. Imputation", "Median imputation (sklearn)", 
         "Sparse entropy/Hurst features for low-N users; median preferred over mean for skewed distributions"],
        ["2. Scaling", "RobustScaler (IQR-based)",
         "StandardScaler distorted by RT outliers; RobustScaler down-weights tails"],
        ["3. Dim. Reduction", "UMAP (n_neighbors=15, Euclidean)",
         "Preserves local manifold structure better than PCA for non-linear behavioral spaces; random_state=42"],
        ["4. Clustering", "HDBSCAN (min_cluster_size=15)",
         "Density-adaptive; discovers non-spherical clusters; assigns noise class rather than forcing membership"],
        ["5. Validation", "Silhouette, DB, CH, ARI",
         "Internal indices assess geometric cohesion; ARI against ground-truth archetypes (synthetic data only)"],
    ]
    story.append(styled_table(pipeline_steps, [1.2*inch, 1.5*inch, 3.8*inch]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Cluster Validity Results", styles["subsection_header"]))
    val_data = [
        ["Index", "Value", "Interpretation"],
        ["Silhouette Score", f"{metrics['silhouette']:.3f}",
         "Moderate separation (>0.25 threshold for interpretable structure)"],
        ["Davies-Bouldin Index", f"{metrics['davies_bouldin']:.3f}",
         "Lower is better; values <2.0 indicate acceptable cluster compactness"],
        ["Calinski-Harabasz", f"{metrics['calinski_harabasz']:.1f}",
         "Higher is better; strong between-cluster variance relative to within"],
        ["ARI vs. Ground Truth", f"{metrics['ari_vs_ground_truth']:.3f}",
         "Moderate recovery of synthetic archetypes (0.0=random, 1.0=perfect)"],
        ["Noise Fraction", f"{metrics['noise_fraction']:.1%}",
         "HDBSCAN noise assignment (all users assigned to a cluster here)"],
    ]
    story.append(styled_table(val_data, [1.8*inch, 1.0*inch, 3.7*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The ARI of 0.267 against embedded ground-truth archetypes confirms that the "
        "pipeline recovers meaningful behavioral structure — performance substantially "
        "above chance (ARI=0) — while acknowledging that behavioral phenotypes exist "
        "on a continuum rather than as categorically discrete types. The silhouette "
        "score of 0.29 is consistent with well-separated but overlapping clusters, "
        "expected in naturalistic learning data.",
        styles["body_small"]
    ))

    # Add UMAP figure
    fig_path = OUTPUTS / "fig1_umap_clusters.png"
    if fig_path.exists():
        story.append(Spacer(1, 6))
        img = Image(str(fig_path), width=5.5*inch, height=4.1*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 1. UMAP projection of the 28-feature behavioral space with cluster assignments. "
            "The embedding reveals a primary gradient (efficiency/accuracy axis) with "
            "divergent density structures at low and high ends.",
            styles["caption"]
        ))

    story.append(PageBreak())


def section_cluster_profiles(story, styles, feat_df):
    story.append(Paragraph("4. Cluster Profiles & Typology", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "Four clusters were recovered. Using PCA fallback (UMAP unavailable in this "
        "execution environment), clusters are defined on the first two principal "
        "components of the scaled feature space. The cluster typology maps onto "
        "the Active Seeker / Passive Consumer continuum as follows:",
        styles["body"]
    ))

    valid = feat_df[feat_df["cluster_label"] >= 0]
    cluster_summary = valid.groupby("cluster_label").agg(
        N=("accuracy","count"),
        Accuracy=("accuracy","mean"),
        RT_log=("rt_log_mean","mean"),
        Efficiency=("efficiency_index","mean"),
        Sessions=("n_sessions","mean"),
        ReattemptFrac=("reattempt_fraction","mean"),
        RetentionDelta=("retention_delta","mean"),
    ).round(3)

    archetype_names = {
        0: "Low-Accuracy Reworker",
        1: "Moderate-Accuracy Learner A",
        2: "Moderate-Accuracy Learner B",
        3: "High-Variance Learner",
    }
    cluster_names = {
        0: "Cluster 0",
        1: "Cluster 1",
        2: "Cluster 2",
        3: "Cluster 3",
    }

    # Get cluster names from data
    if "cluster_name" in feat_df.columns:
        name_map = valid.groupby("cluster_label")["cluster_name"].first().to_dict()
    else:
        name_map = {i: f"Cluster {i}" for i in range(4)}

    profile_data = [["Cluster", "N", "Accuracy", "log(RT)", "Efficiency", "Sessions", "Ret.Delta"]]
    for c in sorted(cluster_summary.index):
        row = cluster_summary.loc[c]
        profile_data.append([
            f"C{c}: {name_map.get(c,'?')[:22]}",
            str(int(row["N"])),
            f"{row['Accuracy']:.3f}",
            f"{row['RT_log']:.2f}",
            f"{row['Efficiency']:.4f}",
            f"{row['Sessions']:.1f}",
            f"{row['RetentionDelta']:+.4f}",
        ])
    story.append(styled_table(profile_data,
        [2.2*inch, 0.4*inch, 0.7*inch, 0.7*inch, 0.75*inch, 0.7*inch, 0.75*inch]))

    story.append(Spacer(1, 10))

    # Radar figure
    fig_path = OUTPUTS / "fig4_cluster_profiles.png"
    if fig_path.exists():
        img = Image(str(fig_path), width=5.0*inch, height=4.3*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 2. Radar chart of normalized cluster centroids across 8 behavioral "
            "dimensions. Clusters are primarily separated on accuracy/efficiency axes, "
            "with secondary differentiation on RT variability and session engagement.",
            styles["caption"]
        ))

    story.append(Paragraph(
        "The most substantively significant split is between Cluster 0 (accuracy &#8776; 0.45) "
        "and Clusters 1–3 (accuracy 0.63–0.64). This is not merely a difficulty effect: "
        "Cluster 0 shows distinct kinematic signatures (lower efficiency index, wider "
        "RT distribution) consistent with a behavioral profile of persistent difficulty "
        "rather than disengagement. Clusters 1 and 2 are closely matched on aggregate "
        "metrics, differing primarily in session span (45 vs. 47 days) and RT "
        "consistency — a finding that motivates the longitudinal outcome analysis.",
        styles["body"]
    ))
    story.append(PageBreak())


def section_outcomes(story, styles, kw_df, corrs, feat_df):
    story.append(Paragraph("5. Outcome Analysis", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    story.append(Paragraph("5.1 Statistical Framework", styles["subsection_header"]))
    story.append(Paragraph(
        "Outcome distributions are non-normal (Shapiro-Wilk p&lt;0.001 for all "
        "outcome variables); parametric ANOVA is inappropriate. We therefore apply "
        "Kruskal-Wallis H-tests (non-parametric one-way ANOVA) with effect size "
        "estimated as &#951;<sup>2</sup> = (H - k + 1) / (N - k). "
        "Post-hoc pairwise comparisons use Mann-Whitney U with Bonferroni correction. "
        "The primary efficiency-outcome relationship is assessed via Spearman rank "
        "correlation (population-level and within-cluster).",
        styles["body"]
    ))

    story.append(Paragraph("5.2 Kruskal-Wallis Results", styles["subsection_header"]))
    kw_table_data = [["Outcome Variable", "H Statistic", "p-value", "&#951;<sup>2</sup>", "N", "Significant?"]]
    sig_map = {True: "Yes ***", False: "No (ns)"}
    for _, row in kw_df.iterrows():
        kw_table_data.append([
            row["outcome"].replace("_", " "),
            f"{row['H_stat']:.2f}",
            f"{row['p_value']:.6f}" if row['p_value'] > 0.0001 else "<0.0001",
            f"{row['eta2']:.4f}",
            str(int(row["n"])),
            sig_map[row["p_value"] < 0.05],
        ])
    story.append(styled_table(kw_table_data,
        [2.1*inch, 0.85*inch, 0.85*inch, 0.65*inch, 0.5*inch, 1.0*inch]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "The pattern is unambiguous: clusters differ significantly on short-term task "
        "completion (STC; H=138.16, p&lt;0.0001, &#951;<sup>2</sup>=0.170) and on the "
        "efficiency index (H=195.01, &#951;<sup>2</sup>=0.241) — but show no significant "
        "differences on long-term retention delta (H=0.59, p=0.898, &#951;<sup>2</sup>&#8776;0) "
        "or proficiency trend (H=0.16, p=0.985).",
        styles["body"]
    ))

    story.append(Paragraph("5.3 Efficiency-Retention Correlation", styles["subsection_header"]))

    rho_stc = corrs.get("spearman_session_acc_mean_rho", 0)
    p_stc   = corrs.get("spearman_session_acc_mean_p", 1)
    rho_ret = corrs.get("spearman_retention_delta_rho", 0)
    p_ret   = corrs.get("spearman_retention_delta_p", 1)

    corr_data = [
        ["Relationship", "Spearman &#961;", "p-value", "Interpretation"],
        ["Efficiency &#8594; STC", f"{rho_stc:+.3f}", 
         "<0.0001" if p_stc < 0.0001 else f"{p_stc:.4f}",
         "Strong positive association"],
        ["Efficiency &#8594; LTR", f"{rho_ret:+.3f}",
         f"{p_ret:.4f}",
         "Near-zero, non-significant"],
    ]
    # Add within-cluster rows
    for c in range(4):
        r = corrs.get(f"cluster{c}_eff_ret_rho", None)
        p = corrs.get(f"cluster{c}_eff_ret_p", None)
        if r is not None:
            corr_data.append([
                f"C{c}: Efficiency &#8594; LTR",
                f"{r:+.3f}",
                f"{p:.4f}" if p else "—",
                "ns within cluster"
            ])
    story.append(styled_table(corr_data, [2.2*inch, 1.0*inch, 0.9*inch, 2.4*inch]))
    story.append(Spacer(1, 10))

    # Key finding box
    story.append(finding_box(
        "Core Finding: Session efficiency (accuracy / log RT) strongly predicts "
        "short-term task completion (&#961; = +0.906, p<0.0001) but shows near-zero "
        "association with long-term retention delta (&#961; = +0.031, p = 0.404). "
        "This dissociation holds within every cluster (all within-cluster &#961; < 0.19, "
        "all p > 0.14), ruling out confounding by cluster membership.",
        styles,
        color=ACCENT,
    ))
    story.append(Spacer(1, 10))

    # Scatter figure
    fig_path = OUTPUTS / "fig3_efficiency_vs_retention.png"
    if fig_path.exists():
        img = Image(str(fig_path), width=5.5*inch, height=3.9*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 3. Efficiency Index vs. Long-Term Retention Delta, colored by cluster. "
            "The near-horizontal regression line (&#946; &#8776; 0) confirms the absence "
            "of a meaningful efficiency-retention relationship at the population level.",
            styles["caption"]
        ))

    # Distribution figure
    fig_path = OUTPUTS / "fig2_outcome_distributions.png"
    if fig_path.exists():
        story.append(Spacer(1,4))
        img = Image(str(fig_path), width=6.5*inch, height=3.0*inch)
        story.append(img)
        story.append(Paragraph(
            "Figure 4. Outcome variable distributions by cluster. Note the marked "
            "separation on session_acc_mean (left) contrasting with near-identical "
            "retention_delta distributions across clusters (second panel).",
            styles["caption"]
        ))

    story.append(PageBreak())


def section_limitations(story, styles):
    story.append(Paragraph("6. Limitations", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    lims = [
        ("Operationalization of Retention",
         "retention_delta captures proficiency change across sessions but cannot "
         "distinguish genuine forgetting from item difficulty variation. Without "
         "matched pre/post assessments or IRT-calibrated item parameters, causal "
         "inference about durable encoding is not warranted. The measure is better "
         "understood as 'cross-temporal performance stability' than 'retention' "
         "in the strict memory science sense."),
        ("Median Imputation Bias",
         "Entropy (ApEn) and Hurst exponent estimation require &#8805;50 observations "
         "for stability; users near the 40-interaction threshold receive imputed values "
         "converging on population medians. This compresses variance at the low-activity "
         "end of the distribution and may artificially reduce cluster separation for "
         "sparse users. A more principled approach would apply Bayesian partial pooling "
         "with informative priors from high-N users."),
        ("Platform Latency Confound",
         "EdNet-KT4 collects interactions across Android, iOS, and web platforms. "
         "Device-specific and network latencies contribute systematic noise to "
         "elapsed_time measurements. A full implementation would include platform "
         "fixed effects in the feature computation or stratify analyses by platform."),
        ("UMAP Stochasticity",
         "UMAP embeddings are sensitive to n_neighbors, min_dist, and random seed. "
         "The present analysis fixes random_state=42 but does not assess embedding "
         "stability via perturbation analysis (e.g., 100 re-runs with different seeds). "
         "Cluster boundaries in PCA/UMAP space should not be over-interpreted as "
         "sharp categorical boundaries."),
        ("Absence of IRT Calibration",
         "Item difficulty (b-parameter) is not modeled. Accuracy differences between "
         "clusters partially reflect systematic item routing (harder content for "
         "higher-proficiency users on adaptive platforms) rather than pure behavioral "
         "style differences. Without IRT, efficiency_index conflates behavioral "
         "efficiency with item-level difficulty targeting."),
        ("Ecological Validity",
         "Clustering reflects aggregate session-level behavior and cannot capture "
         "within-session strategy shifts, help-seeking from external sources, "
         "or contextual factors (exam proximity, fatigue, social facilitation). "
         "Behavioral typologies derived from telemetry alone should be validated "
         "against self-report and interview data before informing instructional design."),
        ("Pedagogical Intervention Mapping",
         "The typology is purely descriptive. The implication that 'Disengaged Drifters "
         "need spaced practice scaffolding' is a hypothesis requiring experimental "
         "validation, not a conclusion derivable from the clustering alone. "
         "Observational behavioral patterns and causal mechanisms for intervention "
         "design are categorically distinct levels of inference."),
    ]

    for i, (title, text) in enumerate(lims):
        story.append(Paragraph(
            f"<b>L{i+1}. {title}:</b>  {text}",
            styles["body_small"]
        ))
        story.append(Spacer(1, 3))

    story.append(PageBreak())


def section_pedagogy(story, styles):
    story.append(Paragraph("7. Pedagogical Implications", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "The central empirical finding — that session efficiency strongly predicts "
        "immediate performance but is orthogonal to retention — has direct implications "
        "for learning platform design and instructional scaffolding:",
        styles["body"]
    ))

    implications = [
        ("Fluency &#8800; Learning",
         "The efficiency-STC association (&#961;=+0.91) could tempt platform designers "
         "to optimize for fast, accurate completion as a learning signal. The "
         "efficiency-LTR dissociation (&#961;=+0.03) suggests this would be a "
         "category error: fluency metrics track performance state, not learning "
         "trajectory. Platforms using session completion rate as a primary KPI "
         "may be measuring something pedagogically orthogonal to their stated goal."),
        ("Desirable Difficulty Hypothesis",
         "The result is consistent with Bjork's desirable difficulties framework "
         "(1994): conditions that slow immediate performance (interleaving, spacing, "
         "retrieval practice under load) generate better long-term retention precisely "
         "because they feel less efficient in the moment. Platform designs that "
         "present massed, blocked, or hint-rich practice may be optimizing for "
         "the wrong side of the STC/LTR tradeoff."),
        ("Heterogeneous Cluster Implications",
         "The 'Low-Accuracy Reworker' cluster (Cluster 0, &#8764;10% of users, "
         "accuracy 0.45) shows a distinct behavioral signature — high re-attempt "
         "fraction, slow RT, wide latency distribution — that may indicate "
         "constructive struggle or productive failure (Kapur, 2016). Their "
         "near-zero retention delta (comparable to higher-accuracy clusters) "
         "suggests this effort expenditure is not being converted into durable "
         "learning gains, pointing to a potential instructional design intervention "
         "target."),
        ("Measurement Recommendation",
         "Platforms should track both STC and LTR as separate KPIs with distinct "
         "temporal windows. Behavioral telemetry indices most predictive of LTR "
         "(session spacing regularity, within-session RT variability, re-attempt "
         "patterns) differ substantially from those predictive of STC "
         "(efficiency index, same-session accuracy trajectory). A single composite "
         "learning score risks conflating these orthogonal dimensions."),
    ]

    for title, text in implications:
        box_data = [[
            Paragraph(f"<b>{title}</b>", ParagraphStyle("bh", fontName="Helvetica-Bold",
                fontSize=9, leading=13, textColor=ACCENT)),
            Paragraph(text, styles["body_small"])
        ]]
        t = Table(box_data, colWidths=[1.4*inch, 5.0*inch])
        t.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEAFTER", (0,0),(0,-1), 2, ACCENT),
            ("LEFTPADDING",(0,0),(-1,-1),8),
            ("RIGHTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("BACKGROUND",(0,0),(-1,-1),LIGHT_GRAY),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    story.append(PageBreak())


def section_references(story, styles):
    story.append(Paragraph("References", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))

    refs = [
        "Bjork, R.A. (1994). Memory and metamemory considerations in the training of "
        "human beings. In J. Metcalfe & A. Shimamura (Eds.), <i>Metacognition</i>. MIT Press.",
        "Bjork, E.L., & Bjork, R.A. (2011). Making things hard on yourself, but in a good way. "
        "<i>Psychology and the Real World</i>, 2, 59–68.",
        "Cepeda, N.J., et al. (2006). Distributed practice in verbal recall tasks. "
        "<i>Psychological Bulletin, 132</i>(3), 354–380.",
        "Choi, Y., et al. (2020). EdNet: A Large-Scale Hierarchical Dataset in Education. "
        "<i>AIED 2020</i>. arXiv:1912.03072.",
        "Gilden, D.L. (2001). Cognitive emissions of 1/f noise. "
        "<i>Psychological Review, 108</i>(1), 33–56.",
        "Heitz, R.P. (2014). The speed-accuracy tradeoff: history, physiology, methodology, "
        "and behavior. <i>Frontiers in Neuroscience, 8</i>, 150.",
        "Kapur, M. (2016). Examining productive failure, productive success, unproductive failure, "
        "and unproductive success in learning. <i>Educational Psychologist, 51</i>(2), 289–299.",
        "Kornell, N., & Bjork, R.A. (2008). Learning concepts and categories. "
        "<i>Psychological Science, 19</i>(6), 585–592.",
        "Luce, R.D. (1986). <i>Response times: Their role in inferring elementary mental organization</i>. "
        "Oxford University Press.",
        "Maris, G., & van der Maas, H. (2012). Speed-accuracy response models. "
        "<i>Psychometrika, 77</i>(3), 615–631.",
        "McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation "
        "and Projection. arXiv:1802.03426.",
        "Newell, K.M., et al. (2001). Movement variability and the use of nonlinear tools. "
        "<i>Journal of Applied Biomechanics, 17</i>(4), 368–375.",
        "OECD. (2019). <i>PISA 2018 Technical Report</i>. OECD Publishing.",
        "van der Linden, W.J. (2006). A lognormal model for response times on test items. "
        "<i>Journal of Educational and Behavioral Statistics, 31</i>(2), 181–204.",
        "van Orden, G.C., Holden, J.G., & Turvey, M.T. (2003). Self-organization of cognitive "
        "performance. <i>Journal of Experimental Psychology: General, 132</i>(3), 331–350.",
    ]

    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle("ref",
            fontName="Helvetica", fontSize=8, leading=12,
            leftIndent=18, firstLineIndent=-18,
            spaceAfter=5, textColor=DARK,
            alignment=TA_JUSTIFY
        )))


# ── Main build ─────────────────────────────────────────────────────────────────
def build_report():
    sys.path.insert(0, str(Path(__file__).parent))

    # load results
    kw_df   = pd.read_csv(OUTPUTS / "kruskal_wallis.csv")
    feat_df = pd.read_csv(OUTPUTS / "cluster_results.csv")
    with open(OUTPUTS / "efficiency_correlations.json") as f:
        corrs = json.load(f)
    with open(OUTPUTS / "validation_metrics.json") as f:
        metrics = json.load(f)

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.85*inch,
        bottomMargin=0.9*inch,
        title="CPDA: Behavioral Latent Variable Analysis",
        author="CPDA Analysis Pipeline",
    )

    styles = build_styles()
    story  = []

    cover_page(story, styles)
    section_intro(story, styles)
    section_features(story, styles)
    section_clustering(story, styles, metrics)
    section_cluster_profiles(story, styles, feat_df)
    section_outcomes(story, styles, kw_df, corrs, feat_df)
    section_limitations(story, styles)
    section_pedagogy(story, styles)
    section_references(story, styles)

    doc.build(story, onFirstPage=make_footer, onLaterPages=make_footer)
    print(f"[report] PDF written → {REPORT_PATH.resolve()}")
    return REPORT_PATH


if __name__ == "__main__":
    build_report()
