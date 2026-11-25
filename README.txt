Case Study 82 - Nimbus Ridge Weather Bulletin Service
Effect of temperature, top-p, max tokens and stop sequences on generated bulletins
LLM, RAG & Agentic AI (CO1)

Team
  Dharsan K            150096724173
  Pratima Vanakhade    150096724149
  Prasad Ganesh Bhad   150096724143
  Krish Bhartiya       150096724166

Contents
  Case_Study_82_Report.docx         Final report: method, results, recommendation, monitoring plan, limitations
  Case_Study_82_Presentation.pptx   Slides (7)
  code/run_experiments.py           API client and experiment runner (400 calls)
  code/analysis.py                  Diversity, length and token-usage analysis and charts
  code/review_flags.py              Faithfulness, truncation and stop-sequence flag analysis and charts
  data/raw_log.csv                  Raw log of all 400 calls (parameters, response text, finish reason, tokens, latency)
  results/                          Summary tables, flag rates, flagged responses, manual review of 20 outputs
  results/charts/                   All figures used in the report

Model: Groq allam-2-7b (free tier).

How to reproduce (run from this folder)
  pip install openai pandas matplotlib
  export API_KEY=... BASE_URL=https://api.groq.com/openai/v1 MODEL=allam-2-7b
  python code/run_experiments.py --workers 2 --delay 2        # writes data/raw_log.jsonl and data/raw_log.csv
  python code/analysis.py --input data/raw_log.csv --output results
  python -I code/review_flags.py data/raw_log.csv results
