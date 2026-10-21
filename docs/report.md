# CA4 result-analysis report

The result-analysis report for CA4 (AI&DS, MGM University, JNEC). It runs to 15 pages and covers:

- introduction and contribution;
- the paper's method and the comparison algorithms;
- the datasets;
- outputs of every stage, six result tables and charts;
- observations, conclusion and references.

[:material-file-pdf-box: Download PDF](assets/generated/CA4_Result_Analysis_Report.pdf){ .md-button .md-button--primary }
[:material-file-word-box: Download Word](assets/generated/CA4_Result_Analysis_Report.docx){ .md-button }

**Group:** Akash Chaudhari (202301128055), Rifa Khan (202301128059), Ayush Shinde (202301128023).

## Rebuilding the report

The report is generated from `results/`, so it always matches the published numbers:

```bash
cd report
python report_charts.py ..                   # summary charts -> results/figures/report_*.png
npm install                                  # docx (npm)
node make_report.js .. CA4_Result_Analysis_Report.docx
soffice --headless --convert-to pdf CA4_Result_Analysis_Report.docx
```

Group members and the observations / conclusion text are in `report/meta.json`.
