import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse
from together import Together
from pygooglenews import GoogleNews
import io

# --- Configuration ---
# Replace with your actual API key for Together AI
together_client = Together(
    api_key="a510758b9bff7bf393548b99848a45972486dd1d699eb86a5e7735d2339c1d8c")

# --- Prompts ---
PROMPT_INDIVIDUAL_ANALYSIS = """Carefully analyze the following news article text for information directly indicating potential **upselling opportunities** specifically for an **employee benefits company in India**. Focus only on details that would suggest a company is growing, investing in employees, or has increased financial capacity to enhance its employee benefits programs.

**Text:**
{provided_text}

Based on your analysis and using the provided categories below, determine the upselling opportunity level and the specific reason(s).

1.  **Opportunity Level (First Line):** State the opportunity level as one of the following:
    * "High Opportunity"
    * "Medium Opportunity"
    * "Low Opportunity"
    * "No Upsell Opportunity Indicated" (If no relevant information is found regarding upselling for an employee benefits company)

2.  **Reason(s) for Opportunity (Second Line):** If an opportunity is indicated, explain the major reason(s) concisely, referencing the relevant category (e.g., "Reason: [Category Name] - Brief explanation."). If there are multiple relevant reasons, list them clearly.

3.  **2-Line Summary of Analysis (Third and Fourth Lines):** Provide a brief, overall summary of the article's relevance to upselling for an employee benefits company, condensing the key findings into exactly two lines. If no upselling opportunity is indicated, summarize why the article is not relevant.

**Categories for Reasons:**
I.  **Financial Growth & Stability:** (Record revenue, profit growth, funding rounds, IPOs, strong financial performance, increased valuation, successful cost optimization leading to increased budgets)
II. **Workforce Expansion & Hiring:** (Mass hiring, significant headcount growth, talent acquisition drives, campus recruitment, leadership hiring, expansion into new markets requiring more employees)
III. **Employee-Centric Initiatives:** (New employee wellness programs, enhanced mental health support, focus on employee experience, DEI initiatives, improved workplace culture, employee recognition programs)
IV. **Strategic Investments & Expansion:** (Business expansion, new product launches, digital transformation, acquisition of new companies, investments in HR tech, major strategic partnerships)
V.  **Market Leadership & Employer Branding:** (Awards for best workplace, "Great Place to Work" recognition, strong employer branding, high employee satisfaction/retention rates)
VI. **Benefits Strategy Evolution:** (Overhaul of benefits, digitization of HR/benefits, adoption of new benefits platforms, focus on flexible/hybrid work benefits, efforts to optimize benefits structure)
VII. **Compliance & Regulatory Readiness:** (Proactive measures for new labor laws, tax benefits for employees, social security code adherence, ensuring comprehensive benefits compliance)

**Example Output Format (for High/Medium/Low Opportunity):**
High Opportunity
Reason: Financial Growth & Stability - Company announced record profits and significant investment plans, suggesting increased budget for employee benefits.
Summary: The company's robust financial performance indicates a strong capacity and potential willingness to invest more in comprehensive employee benefits. This presents a prime opportunity for enhanced offerings.

**Example Output Format (for No Opportunity):**
No Upsell Opportunity Indicated
Summary: The article discusses general industry trends not specific to the company's growth or employee-related initiatives. It provides no indication of changes relevant to enhanced employee benefits or potential upselling.
"""

PROMPT_COMBINED_ANALYSIS = """Given the individual analyses of news articles related to a company and potential client upselling opportunities, provide an overall summary (at most 4 lines).

**Individual Article Analyses:**
{individual_analyses_summary}

In the first line, state the overall opportunity level for upselling for the company (e.g., "Overall High Opportunity," "Overall Medium Opportunity," "Overall Low Opportunity," "Overall No Upsell Opportunity Indicated"). In the subsequent lines, summarize the major reasons for this overall opportunity, drawing from the categories mentioned in the individual analyses. Be concise and focus on the most impactful reasons across all articles. If no relevant information is found across all articles, state "Overall No Upsell Opportunity Indicated."
"""


# --- Functions ---


# Cache results for 1 hour to avoid repeated API calls
@st.cache_data(ttl=3600)
def analyze_text(company_name, provided_text, prompt_template, _together_client):
    """Analyzes the provided text for upselling indicators using Together AI."""
    prompt = prompt_template.format(
        company_name=company_name, provided_text=provided_text)
    try:
        response = _together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content
        return output if output else f"Unexpected response: {output}"
    except Exception as e:
        st.error(f"Error querying Together AI for {company_name}: {e}")
        return "Analysis failed due to AI service error."


@st.cache_data(ttl=3600)  # Cache news fetching for 1 hour
def fetch_news(company_name, from_date, to_date, max_articles=10, queries=None, allowed_domains=None):
    """
    Fetches news articles for a given company using the pygooglenews library.
    Filters articles by allowed domains.
    """
    gn = GoogleNews(lang='en', country='IN')
    results = []
    if queries is None:
        queries = [company_name]

    try:
        # Process queries in groups of 3 to optimize API calls
        for i in range(0, len(queries), 3):
            group_queries = queries[i:i+3]
            combined_query = " OR ".join(group_queries)
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            search_results = gn.search(
                combined_query, from_=from_date_str, to_=to_date_str)

            if search_results and 'entries' in search_results:
                articles_for_query = []
                if allowed_domains:
                    for article in search_results['entries']:
                        source_link = article.get('source', {}).get('href', '')
                        parsed_uri = urlparse(source_link)
                        domain = parsed_uri.netloc.replace('www.', '')
                        if any(d in domain for d in allowed_domains):
                            articles_for_query.append(article)
                    # If no articles from allowed domains, add the top article as a fallback
                    if not articles_for_query and search_results['entries']:
                        articles_for_query.append(search_results['entries'][0])
                else:
                    articles_for_query = search_results['entries']

                results.extend(articles_for_query[:max_articles])
            else:
                st.warning(
                    f"No results or 'entries' not found for query '{combined_query}'")
    except Exception as e:
        st.error(f"Error fetching news for {company_name}: {e}")
        return None
    # Ensure total articles returned is at most max_articles
    return results[:max_articles]


def process_article(article):
    """Extracts summary or title from a news article."""
    return article.get('summary') or article.get('title') or ""


def analyze_news(company_name, from_date, to_date, max_articles=10, queries=None, allowed_domains=None):
    """
    Fetches news articles for a company and analyzes them for upselling indicators.
    """
    st.subheader(f"Analyzing News for **{company_name}**")
    all_articles = fetch_news(company_name, from_date,
                              to_date, max_articles, queries, allowed_domains)

    if not all_articles:
        return {"individual_analyses": [], "overall_summary": "No relevant news articles found for analysis."}

    individual_analyses_list = []
    combined_analysis_text_for_model = ""

    for i, article in enumerate(all_articles):
        article_text = process_article(article)
        article_url = article.get('link', 'No URL available')
        # Get actual title or fallback
        article_title = article.get('title', f"Article {i+1}")

        if article_text:
            analysis_result = analyze_text(
                company_name, article_text, PROMPT_INDIVIDUAL_ANALYSIS, together_client)
            individual_analyses_list.append({
                "title": article_title,  # Store the title
                "url": article_url,
                "analysis": analysis_result
            })
            combined_analysis_text_for_model += f"Article {i+1} Analysis:\n{analysis_result}\n\n"
        else:
            no_text_analysis = "No Upsell Opportunity Indicated (No text in article summary/title)."
            individual_analyses_list.append({
                "title": article_title,  # Store the title even if no text
                "url": article_url,
                "analysis": no_text_analysis
            })
            combined_analysis_text_for_model += f"Article {i+1} Analysis:\n{no_text_analysis}\n\n"

    overall_summary_result = "Overall No Upsell Opportunity Indicated."
    if individual_analyses_list:
        combined_prompt = PROMPT_COMBINED_ANALYSIS.format(
            individual_analyses_summary=combined_analysis_text_for_model.strip())
        overall_summary_result = analyze_text(
            company_name, combined_prompt, "{provided_text}", together_client)

    return {"individual_analyses": individual_analyses_list, "overall_summary": overall_summary_result}


def get_opportunity_level(summary_text):
    """Extracts opportunity level from a summary string."""
    summary_text_lower = summary_text.lower()
    if "low opportunity" in summary_text_lower:
        return "Low Opportunity"
    elif "medium opportunity" in summary_text_lower:
        return "Medium Opportunity"
    elif "high opportunity" in summary_text_lower:
        return "High Opportunity"
    elif "no upsell opportunity indicated" in summary_text_lower:
        return "No Upsell Opportunity Indicated"
    return "Unknown Opportunity"  # Fallback


def display_summary_with_color(company_name, summary_text):
    """Displays the summary with color coding based on opportunity level."""
    opportunity_level = get_opportunity_level(summary_text)

    st.markdown(f"### Summary for {company_name}")  # New heading format

    if "High Opportunity" in opportunity_level:
        st.success(summary_text)  # Green for high opportunity
    elif "Medium Opportunity" in opportunity_level:
        st.info(summary_text)  # Blue for medium opportunity
    elif "Low Opportunity" in opportunity_level:
        st.warning(summary_text)  # Orange for low opportunity
    else:
        # For "No Upsell Opportunity Indicated" and "Unknown Opportunity"
        st.error(summary_text)  # Red for no/unknown opportunity


def run_analysis(company_names, days_to_search):
    """Main function to orchestrate the news fetching and analysis for multiple companies."""
    results = {}
    today = datetime.today()
    # Use user-inputted days for the date range
    from_date = today - timedelta(days=days_to_search)
    max_articles_per_query = 10

    # --- YOUR SPECIFIED UPSELL KEYWORDS ---
    upsell_keywords = [
        # Original
        "employee wellness", "mental health", "gym memberships", "lifestyle benefits",
        "remote work policy", "hybrid work", "flexible hours", "wellness programs",
        "employee upskilling", "learning program", "L&D initiatives", "digital training",
        "career development", "skill building", "internal promotions",
        "diversity equity inclusion", "LGBTQ+ policy", "gender affirmation",
        "disability inclusion", "inclusive benefits", "DEI initiatives",
        "employer of choice", "employee engagement strategy", "workplace culture",
        "employee satisfaction", "employee retention", "great place to work",
        "benefits automation", "HR tech", "employee benefits platform", "benefits outsourcing",
        "AI in HR", "HRMS integration", "benefits digitization",
        "labor law compliance", "social security code", "employee tax benefits",
        "fringe benefit tax", "benefits structure optimization",
        "mass hiring", "talent acquisition strategy", "hiring surge", "hiring spree",
        "campus recruitment", "talent war", "employer branding",

        # New additions: Financial & Business Growth
        "record revenue", "profit growth", "EBITDA margin increase", "business expansion",
        "funding round", "Series A funding", "Series B funding", "IPO plans",
        "profit surge", "quarterly growth", "yearly growth", "financial turnaround",
        "cash flow positive", "valuation increase", "market share gain",
        "cost optimization success",

        # Hiring & Organizational Growth
        "hiring plans", "expansion hiring", "workforce expansion", "headcount growth",
        "employee growth", "talent acquisition drive", "leadership hiring",

        # Strategic Initiatives
        "business transformation", "employee experience initiative", "workplace digitization",
        "employee-first culture", "future of work", "benefits overhaul", "HR transformation",
        "employee engagement program", "rewards and recognition",

        # Market Recognition & Awards
        "best workplace award", "great place to work", "employer of the year",
        "HR excellence award", "diversity champion"
    ]

    # --- YOUR SPECIFIED ALLOWED DOMAINS (UNCHANGED) ---
    allowed_domains = [
        "livemint.com", "economictimes.indiatimes.com", "business-standard.com",
        "thehindubusinessline.com", "financialexpress.com", "ndtvprofit.com",
        "zeebiz.com", "moneycontrol.com", "bloombergquint.com",
        "cnbctv18.com", "businesstoday.in", "indianexpress.com",
        "thehindu.com", "reuters.com", "businesstraveller.com",
        "sify.com", "telegraphindia.com", "outlookindia.com",
        "firstpost.com", "pulse.zerodha.com", "ndtvprofit.com",
        "ddnews.gov.in", "newsonair.gov.in", "pib.gov.in",
        "niti.gov.in", "rbi.org.in", "sebi.gov.in",
        "dpiit.gov.in", "investindia.gov.in", "indiabriefing.com",
        "Taxscan.in", "bwbusinessworld.com", "inc42.com",
        "yourstory.com", "vccircle.com", "entrackr.com",
        "the-ken.com", "linkedin.com", "mca.gov.in",
        "zaubacorp.com", "tofler.in"
    ]

    processed_allowed_domains = [domain.replace(
        "www.", "") for domain in allowed_domains]

    st.sidebar.subheader("Analysis Parameters")
    st.sidebar.info(
        f"Analyzing news from: **{from_date.strftime('%Y-%m-%d')}** to **{today.strftime('%Y-%m-%d')}** ({days_to_search} days)")
    st.sidebar.info(f"Max Articles per Query: **{max_articles_per_query}**")
    st.sidebar.info(
        f"Filtered by {len(processed_allowed_domains)} specified business news domains.")

    for company in company_names:
        queries = [company] + \
            [f"{company} {keyword}" for keyword in upsell_keywords]
        company_analysis = analyze_news(
            company, from_date, today, max_articles_per_query, queries, processed_allowed_domains
        )
        results[company] = company_analysis if company_analysis else {
            "overall_summary": "Analysis failed."}
    return results


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Company Upsell Opportunity Analyzer", layout="wide")
st.title("📈 Company Upsell Opportunity Analysis (India Focus)")
st.markdown("""
This application helps identify potential **upselling opportunities** for an employee benefits company in India by analyzing recent news articles.
When companies are hiring more people or doing very well financially, there might be opportunities to upsell employee benefits products.
Upload an Excel file containing company names, or enter them manually, and the app will fetch relevant news, summarize it, and provide upselling opportunity assessments.
""")

# User input for number of days
days_to_search = st.slider(
    "Select the number of days for news search (back from today):",
    min_value=1,
    max_value=365,
    value=90,  # Default to 90 days
    step=1,
    help="This determines how far back in time the news articles will be fetched."
)

# File uploader widget - MODIFIED TO ACCEPT EXCEL FILES
uploaded_file = st.file_uploader(
    "Upload your 'company_names.xlsx' or 'company_names.xls' file", type=["xlsx", "xls"])

company_names_from_upload = []
if uploaded_file is not None:
    try:
        # MODIFIED TO READ EXCEL FILE
        company_df = pd.read_excel(uploaded_file)
        if "CompanyName" in company_df.columns:
            company_names_from_upload = company_df["CompanyName"].dropna(
            ).tolist()
            if company_names_from_upload:
                st.success(
                    f"Successfully loaded **{len(company_names_from_upload)}** companies from **'{uploaded_file.name}'**.")
                st.info("You can now click 'Start Analysis' to begin.")
            else:
                st.warning(
                    "The 'CompanyName' column is empty after loading. Please check your Excel file.")
        else:
            st.error(
                "Error: The uploaded Excel file must contain a **'CompanyName'** column.")
    except Exception as e:
        st.error(
            f"Error reading Excel file: {e}. Please ensure it's a valid Excel file with the correct column name.")
else:
    st.info("Please upload an Excel file with a 'CompanyName' column to proceed.")

# Custom input for company names (optional fallback)
st.markdown("---")
st.subheader("Or enter company names manually (comma-separated):")
manual_company_input = st.text_input(
    "Example: Reliance Industries, Tata Consultancy Services, Wipro")
if manual_company_input:
    manual_companies = [c.strip()
                        for c in manual_company_input.split(',') if c.strip()]
    if manual_companies:
        st.info(
            f"Using manually entered companies: **{', '.join(manual_companies)}**")
        company_names_to_analyze = manual_companies
    else:
        company_names_to_analyze = company_names_from_upload
else:
    company_names_to_analyze = company_names_from_upload


if st.button("🚀 Start Analysis"):
    if not company_names_to_analyze:
        st.warning(
            "No company names available for analysis. Please upload an Excel file or enter names manually.")
    else:
        with st.spinner("Crunching numbers and fetching news... This might take a while for each company."):
            analysis_results = run_analysis(
                company_names_to_analyze, days_to_search)  # Pass days_to_search

        st.success("🎉 Analysis Complete!")
        st.markdown("---")

        # Display the results
        for company, analysis in analysis_results.items():
            st.markdown(f"## :office: {company}")

            # Display Overall Upsell Opportunity Summary with color and new heading
            display_summary_with_color(company, analysis.get(
                "overall_summary", "No overall analysis available."))

            st.markdown("### Individual Article Analyses")
            if analysis.get("individual_analyses"):
                for i, article_analysis in enumerate(analysis["individual_analyses"]):
                    # Display actual article title
                    st.markdown(
                        f"#### :newspaper: {article_analysis['title']}")
                    st.markdown(f"**URL:** [Link]({article_analysis['url']})")
                    article_analysis_text = article_analysis['analysis']
                    # Color-code individual analysis based on opportunity
                    if "High Opportunity" in article_analysis_text:
                        st.success(f"**Analysis:** {article_analysis_text}")
                    elif "Medium Opportunity" in article_analysis_text:
                        st.info(f"**Analysis:** {article_analysis_text}")
                    elif "Low Opportunity" in article_analysis_text:
                        st.warning(f"**Analysis:** {article_analysis_text}")
                    else:
                        st.error(f"**Analysis:** {article_analysis_text}")
                    st.markdown("---")
            else:
                st.info("No individual articles found for detailed analysis.")
            st.markdown("---")  # Separator between companies

        # Export to Excel
        data_for_df = []
        for company, analysis in analysis_results.items():
            overall_summary = analysis.get(
                "overall_summary", "No analysis available")
            overall_opportunity_level = get_opportunity_level(
                overall_summary)  # Get overall opportunity level

            company_data = {
                "Company": company,
                # New column for opportunity level
                "Overall Opportunity Level": overall_opportunity_level,
                "Overall Summary": overall_summary
            }
            for i, article_analysis in enumerate(analysis.get("individual_analyses", [])):
                # Get individual article opportunity level
                article_opportunity_level = get_opportunity_level(
                    article_analysis["analysis"])
                company_data[f"Article {i+1} Title"] = article_analysis["title"]
                company_data[f"Article {i+1} URL"] = article_analysis["url"]
                # Add opportunity level for individual article
                company_data[f"Article {i+1} Opportunity Level"] = article_opportunity_level
                company_data[f"Article {i+1} Analysis"] = article_analysis["analysis"]
            data_for_df.append(company_data)

        if data_for_df:
            df_results = pd.DataFrame(data_for_df)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_file_name = f"upsell_analysis_results_{timestamp}.xlsx"

            excel_buffer = io.BytesIO()
            df_results.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            excel_buffer.seek(0)

            st.download_button(
                label="Download All Results as Excel 📊",
                data=excel_buffer,
                file_name=excel_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Click to download the comprehensive analysis results."
            )
            st.success("Results are ready for download!")
        else:
            st.warning(
                "No data to export to Excel, as no analysis was performed.")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app leverages Together AI's Llama-3.3-70B-Instruct-Turbo-Free model and Google News for upselling opportunity analysis.")
