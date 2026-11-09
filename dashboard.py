"""Churn Analytics — run from the customer-churn-platform project root."""
from pathlib import Path
import html
import json
import os
import math
from datetime import datetime, timezone
from uuid import uuid4
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
API = os.getenv('CHURN_API_URL', 'https://isd1yxfjpi.execute-api.ap-southeast-2.amazonaws.com').rstrip('/')
st.set_page_config(page_title='Customer Churn & Revenue Risk Analytics', page_icon='◈', layout='wide')
st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at 85% 0%,#7c3aed18,transparent 45%),#090b12;color:#e8eaf3}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.3rem;padding-bottom:3rem;max-width:1500px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#111021,#0c1020);border-right:1px solid #2b2443}
[data-testid="stSidebar"] *{color:#dcd9ee}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:12px;border-radius:12px;background:#181528;border:1px solid #2c2541;margin-bottom:6px}
h1,h2,h3{letter-spacing:-.035em;color:#f2efff!important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:#aaa4be!important}
.hero{background:linear-gradient(115deg,#171126,#2b1750 60%,#182039);border:1px solid #624391;border-radius:22px;padding:32px 36px;margin:10px 0 25px;overflow:hidden}
.hero h1{font-size:36px;margin:8px 0 12px;line-height:1.18;max-width:850px}
.hero p{color:#c5bddb;max-width:800px;font-size:15px;margin:0}
.eyebrow{font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:700;color:#67e8f9}
.brand{font-size:25px;letter-spacing:-1px;font-weight:750;color:white}.brand span{color:#b794ff}
.small{font-size:12px;color:#a19ab7;line-height:1.7}
[data-testid="stMetric"]{background:linear-gradient(135deg,#191525,#121522);border:1px solid #34284c;border-top:3px solid #a78bfa;border-radius:16px;padding:22px}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"]{border-top-color:#c084fc}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"]{border-top-color:#67e8f9}
[data-testid="stMetricValue"]{font-size:30px;color:#f5f0ff!important}
[data-testid="stMetricLabel"]{color:#b9b3cf!important}
[data-testid="stForm"]{background:#111420!important;border:1px solid #403052;border-radius:20px;padding:26px}
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *,[data-testid="stForm"] label p{color:#eee8ff!important;opacity:1!important}
[data-testid="stForm"] h3{color:#c4b5fd!important;border-bottom:1px solid #34284c;padding-bottom:12px}
[data-baseweb="input"],[data-baseweb="input"] input,[data-baseweb="select"]>div{background:#1b1d2e!important;color:#f1edff!important;border-color:#403456!important;border-radius:10px}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="popover"] [role="option"]{background:#1b1d2e!important;color:#f5f0ff!important}
[data-baseweb="popover"] [role="option"]:hover{background:#35244f!important}
.stFormSubmitButton>button[kind="primary"],.stButton>button[kind="primary"]{background:linear-gradient(105deg,#7c3aed,#a855f7)!important;color:white!important;border:1px solid #b794ff;border-radius:12px;min-height:46px;box-shadow:0 6px 22px #7c3aed30}
.stDownloadButton>button{color:#d8c7ff!important;background:#211733!important;border:1px solid #7c5aa8;border-radius:10px}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{border:1px solid #332a47;border-radius:16px;overflow:hidden}
.footer{font-size:12px;color:#9690ab;border-top:1px solid #30273f;padding-top:18px;margin-top:30px}
.report-section{background:#121522;border:1px solid #322b45;border-radius:16px;padding:25px;margin:16px 0;color:#e8eaf3}
.report-section h3{margin:5px 0 15px;font-size:22px}
.report-section p{color:#c4bfd4;line-height:1.75;margin:8px 0}
.report-section .pill{display:inline-block;border:1px solid #685185;background:#2c2141;border-radius:30px;padding:5px 12px;color:#ddceff;font-size:12px;margin:3px}
.report-section .pill.alert{border-color:#896b35;background:#382b19;color:#ffe0a1}
.report-section .pill.calm{border-color:#2d6a63;background:#16312e;color:#a5eee0}
.report-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.report-fact{background:#191a2c;padding:16px;border-radius:10px;overflow-wrap:anywhere}
.report-fact small{display:block;color:#aaa3c0;font-size:12px;margin-bottom:7px}
.report-fact strong{color:#f5f0ff;font-size:16px}
.action-row{display:flex;gap:15px;border-top:1px solid #30283f;padding:16px 0}
.action-number{color:#c4b5fd;font-size:22px;font-weight:700;min-width:32px}
.action-row h4{margin:0;color:#eee8ff;font-size:16px}
.action-row p{margin:6px 0 0;font-size:14px}
.report-note{border-left:3px solid #67e8f9;padding:12px 18px;background:#14212d;color:#d2e9f2;line-height:1.7;margin:14px 0;border-radius:0 10px 10px 0}
@media(max-width:700px){.report-grid{grid-template-columns:1fr}.hero{padding:24px}.hero h1{font-size:28px}}
</style>""", unsafe_allow_html=True)

def chart(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=15, r=15, t=35, b=20),
        paper_bgcolor="#111420",
        plot_bgcolor="#111420",
        font=dict(family="Arial", color="#c8c2dd"),
        colorway=["#a78bfa", "#67e8f9", "#f0abfc", "#fbbf24"],
        legend=dict(orientation="h", y=-0.16),
        title_font_size=15,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#28243a", zeroline=False)

    # Match the prediction gauge to the dark background.
    for trace in fig.data:
        if trace.type == "indicator":
            trace.number.font.color = "#f5f0ff"
            trace.gauge.bgcolor = "#252039"
            trace.gauge.axis.tickfont.color = "#bcb4d2"
            trace.gauge.threshold.line.color = "#67e8f9"
            for step in trace.gauge.steps:
                step.color = "#252039"

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
    )

def hero(kicker,title,description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></div>',unsafe_allow_html=True)

def read_csv(relative):
    path=ROOT/relative
    if not path.exists(): return None
    try: return pd.read_csv(path)
    except Exception as exc:
        st.error(f'Could not read {relative}: {exc}')
        return None

def validate_prediction(customer, result):
    """Reject malformed or inconsistent responses before publishing a report."""
    probability = float(result['ChurnProbability'])
    threshold = float(result['Threshold'])
    exposure = float(result['RevenueExposureScore'])
    if not all(math.isfinite(v) for v in (probability, threshold, exposure)):
        raise ValueError('Non-finite prediction value')
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1 or exposure < 0:
        raise ValueError('Prediction values out of range')
    if result['PredictedChurn'] not in (0, 1):
        raise ValueError('Invalid classification')
    if int(result['PredictedChurn']) != int(probability >= threshold):
        raise ValueError('Classification does not match threshold')
    if str(result['customerID']) != customer['customerID']:
        raise ValueError('Customer ID does not match request')
    if not math.isclose(exposure, probability * customer['MonthlyCharges'], rel_tol=1e-5, abs_tol=0.001):
        raise ValueError('Exposure calculation does not match')


def report_content(customer, result, meta):
    """One source of content for the on-screen and downloaded reports."""
    esc = lambda value: html.escape(str(value), quote=True)
    p = float(result['ChurnProbability'])
    threshold = float(result['Threshold'])
    exposure = float(result['RevenueExposureScore'])
    flagged = bool(result['PredictedChurn'])
    gap = (p - threshold) * 100
    status = 'Flagged for retention review' if flagged else 'Below the review threshold'
    if abs(gap) < 0.005:
        comparison = 'equal to the classification threshold'
    else:
        comparison = f'{abs(gap):.2f} percentage points {"above" if gap > 0 else "below"} the classification threshold'
    summary = (
        f"The model estimates a {p:.2%} churn probability for this submitted customer profile. "
        f"The score is {comparison} of {threshold:.2%}. "
        + ('This profile is therefore flagged for a retention review. A conversation with the customer can help assess whether support is needed.'
           if flagged else 'This profile is therefore not flagged under the current decision rule. Continue routine service monitoring; a score below the threshold does not guarantee retention.')
    )
    service_names = {'OnlineSecurity':'Online security','OnlineBackup':'Online backup','DeviceProtection':'Device protection','TechSupport':'Tech support','StreamingTV':'Streaming TV','StreamingMovies':'Streaming movies'}
    active = [label for key,label in service_names.items() if customer[key] == 'Yes']
    if customer['InternetService'] == 'No':
        service_description = 'No internet service is recorded, so internet add-ons are not applicable.'
    else:
        service_description = f"{len(active)} of six internet add-ons are subscribed. " + ('Selected services: ' + ', '.join(active) + '.' if active else 'No internet add-ons are selected.')
    profile_note = (
        f"The submitted profile records {customer['tenure']} months of tenure, "
        f"a {customer['Contract'].lower()} contract, and monthly charges of {customer['MonthlyCharges']:,.2f}. "
        + service_description
    )
    if flagged:
        actions = [
            ('Review the account', 'Check recent support interactions and billing concerns before contacting the customer. These details are not available to this model.'),
            ('Start a targeted conversation', 'Ask about service satisfaction, unresolved issues and changing needs. Use the score to prioritize a review, not to assume a reason for leaving.'),
            ('Choose and measure an intervention', 'If a need is confirmed, consider an appropriate service resolution or offer under business policy. Record the action and later outcome; savings are not estimated here.'),
        ]
    else:
        actions = [
            ('Continue routine monitoring', 'Keep standard service and satisfaction checks in place. This classification is not a guarantee that the customer will remain.'),
            ('Respond to new evidence', 'Investigate complaints or account changes when they occur. Operational evidence can justify a review even when the model score is below the threshold.'),
            ('Refresh the customer profile', 'Re-score when relevant customer details change and record the new result. Avoid interpreting an old prediction as a permanent customer status.'),
        ]
    facts = [
        ('Customer ID', customer['customerID']), ('Tenure', f"{customer['tenure']} months"),
        ('Contract', customer['Contract']), ('Payment method', customer['PaymentMethod']),
        ('Monthly charges', f"{customer['MonthlyCharges']:,.2f}"), ('Total charges', f"{customer['TotalCharges']:,.2f}"),
        ('Internet service', customer['InternetService']), ('Phone service', customer['PhoneService']),
        ('Multiple lines', customer['MultipleLines']), ('Paperless billing', customer['PaperlessBilling']),
        ('Gender', customer['gender']), ('Senior citizen', 'Yes' if customer['SeniorCitizen'] else 'No'),
        ('Partner', customer['Partner']), ('Dependents', customer['Dependents']),
    ]
    def section(number, title, body):
        return f'<section class="report-section"><div class="eyebrow">{number}</div><h3>{title}</h3>{body}</section>'
    header = section('CHURN ANALYTICS / CUSTOMER ASSESSMENT', 'Customer Risk Assessment Report',
        f'<p><strong>{esc(customer["customerID"])}</strong> · {esc(meta["report_id"])}</p>'
        f'<p>Generated {esc(meta["generated_at"])}</p>'
        f'<span class="pill {"alert" if flagged else "calm"}">{status}</span><span class="pill">Single-customer assessment</span>')
    executive = section('01 / EXECUTIVE SUMMARY', 'Assessment Summary', f'<p>{esc(summary)}</p>')
    profile = section('02 / CUSTOMER CONTEXT', 'Customer Profile',
        f'<p>{esc(profile_note)}</p><div class="report-grid">' + ''.join(
            f'<div class="report-fact"><small>{esc(label)}</small><strong>{esc(value)}</strong></div>' for label,value in facts
        ) + '</div><p>Subscribed internet add-ons</p>' + (
            ''.join(f'<span class="pill">{esc(label)}</span>' for label in active)
            if active else '<span class="pill">None / not applicable</span>'
        ) + '<p class="small">These are submitted facts, not verified causes of the model score.</p>')
    revenue = section('03 / REVENUE CONTEXT', 'Revenue Exposure Analysis',
        f'<p>Churn probability × monthly charges</p><h3>{p:.4f} × {customer["MonthlyCharges"]:,.2f} ≈ {exposure:,.2f}</h3>'
        '<p>The displayed calculation is rounded; the API uses the full probability. Monetary values retain the source dataset’s units.</p>'
        '<div class="report-note">Use this score to compare and prioritize customer reviews. It is not confirmed lost revenue, expected savings, or a forecast for a defined future period. Probability calibration has not been established.</div>')
    followup = section('04 / ACTION PLAN', 'Recommended Follow-up Actions',
        '<p>General workflow guidance based on the flag. These actions are not model explanations or proven treatments.</p>' + ''.join(
            f'<div class="action-row"><div class="action-number">{index:02d}</div><div><h4>{esc(title)}</h4><p>{esc(body)}</p></div></div>'
            for index,(title,body) in enumerate(actions,1)))
    evidence = section('05 / MODEL CONTEXT', 'Model Evaluation & Limitations',
        '<p><strong>Project model:</strong> tuned Gradient Boosting. Recorded evaluation used 1,409 held-out customers: ROC-AUC 0.8457, recall 71.66%, precision 56.90%, and F1 0.6343 at threshold 0.35.</p>'
        '<p>These are aggregate project results, not confidence estimates for this customer. The endpoint does not return a model version identifier, so these metrics are not independently verified for each API response.</p>'
        '<p>The threshold was selected using training out-of-fold F1. Initial EDA used the full dataset; tuning and threshold selection reused training data and were not nested.</p>'
        '<div class="report-note">Individual SHAP explanations are not returned by this endpoint. Customer-specific drivers and causal reasons are therefore not claimed in this report.</div>')
    return dict(header=header, executive=executive, profile=profile, revenue=revenue, followup=followup, evidence=evidence,
                status=status, summary=summary, gap=gap)


def downloadable_report(customer, result, meta, sections):
    """Self-contained HTML, printable through the browser without dependencies."""
    esc = lambda value: html.escape(str(value), quote=True)
    p = float(result['ChurnProbability']); threshold = float(result['Threshold'])
    css = '''
    :root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#090b12;color:#eee8ff;font:15px/1.65 Arial,sans-serif}
    main{max-width:1080px;margin:auto;padding:32px}h1,h2,h3,h4{line-height:1.3}h3{font-size:23px;margin:7px 0 14px}p{color:#c6bfd8}
    .eyebrow{font-size:10px;font-weight:700;letter-spacing:2px;color:#67e8f9}.report-section{background:#121522;border:1px solid #342a48;border-radius:16px;padding:26px;margin:20px 0;overflow-wrap:anywhere}
    .report-section:first-of-type{background:linear-gradient(110deg,#211234,#17253a);border-color:#76579f}.pill{display:inline-block;font-size:12px;border:1px solid #685185;background:#2c2141;color:#ddceff;border-radius:30px;padding:5px 12px;margin:3px}.alert{color:#ffe0a1;background:#382b19;border-color:#896b35}.calm{color:#a5eee0;background:#16312e;border-color:#2d6a63}
    .report-grid,.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.report-fact,.metric{background:#1a192b;border-radius:12px;padding:17px;overflow-wrap:anywhere}.report-fact small,.metric small{display:block;color:#b0a6c7;margin-bottom:5px}.metric strong{font-size:28px;color:#e4d9ff}.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.metric{border-top:3px solid #a78bfa}
    .action-row{display:flex;gap:16px;padding:17px 0;border-top:1px solid #342a48}.action-number{font-size:24px;font-weight:bold;color:#c4b5fd}.action-row h4{margin:0}.action-row p{margin:7px 0}.report-note{border-left:3px solid #67e8f9;background:#15222e;padding:14px 18px;color:#d4e9f4;margin:14px 0}.small,footer{font-size:12px;color:#a69dbb}
    .meter{height:14px;background:#292139;border-radius:8px;position:relative;margin:18px 0}.fill{height:100%;background:linear-gradient(90deg,#7c3aed,#c084fc);border-radius:8px}.threshold{position:absolute;top:-5px;height:24px;width:3px;background:#67e8f9}.print-help{text-align:right;color:#bdb4d1;font-size:12px}
    @media(max-width:650px){main{padding:14px}.metrics,.report-grid{grid-template-columns:1fr}}
    @media print{:root{color-scheme:light}@page{size:A4;margin:16mm}body{background:white;color:#17233a;font-size:10pt}main{padding:0;max-width:none}.print-help{display:none}.report-section,.report-section:first-of-type{background:white;border:1px solid #cdd2df;padding:16px;border-radius:8px}.report-fact,.metric{background:#f3f1f8}.report-fact small,.metric small,p,.small,footer{color:#475569}.metric strong,h3,h4{color:#30204e}.eyebrow{color:#624092}.pill,.pill.alert,.pill.calm{background:#f3f1f8;color:#30204e;border-color:#c8bdd7}.report-note{background:#f1f6fa;color:#24394f}.action-row,.report-fact,.metric{break-inside:avoid}h3,h4{break-after:avoid}.meter,.fill{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
    '''
    metrics = f'''<section class="report-section"><div class="eyebrow">PREDICTION METRICS</div><div class="metrics"><div class="metric"><small>Churn probability</small><strong>{p:.2%}</strong></div><div class="metric"><small>Decision threshold</small><strong>{threshold:.2%}</strong></div><div class="metric"><small>Exposure score</small><strong>{float(result['RevenueExposureScore']):,.2f}</strong></div></div><div class="meter"><div class="fill" style="width:{p*100:.4f}%"></div><div class="threshold" style="left:{threshold*100:.4f}%"></div></div><p>Purple fill: model probability. Cyan marker: classification threshold. Flagged when probability is greater than or equal to the threshold.</p></section>'''
    content=sections['header']+metrics+''.join(sections[key] for key in ['executive','profile','revenue','followup','evidence'])
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Customer risk report — {esc(customer["customerID"])}</title><style>{css}</style></head><body><main><div class="print-help">Print / save as PDF: press Ctrl+P in your browser.</div>{content}<footer>Churn Analytics · Portfolio demonstration · {esc(meta["report_id"])}</footer></main></body></html>'


def render_customer_report(customer, result, meta):
    sections=report_content(customer,result,meta)
    p=float(result['ChurnProbability']); threshold=float(result['Threshold'])
    st.divider()
    st.caption('This report describes the last submitted profile. Submit again to include any edits to the form.')
    st.markdown(sections['header'],unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric('Churn probability',f'{p:.2%}')
    b.metric('Decision threshold',f'{threshold:.2%}')
    c.metric('Illustrative exposure',f'{float(result["RevenueExposureScore"]):,.2f}')
    l,r=st.columns([1,1.65])
    with l:
        fig=go.Figure(go.Indicator(mode='gauge+number',value=p*100,number={'suffix':'%','valueformat':'.2f','font':{'size':42}},title={'text':'Probability vs. threshold'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#a78bfa'},'steps':[{'range':[0,100],'color':'#252039'}],'threshold':{'line':{'color':'#67e8f9','width':3},'value':threshold*100}}))
        chart(fig,285)
        st.caption('Purple: model score · Cyan marker: decision threshold')
    with r: st.markdown(sections['executive'],unsafe_allow_html=True)
    tabs=st.tabs(['Customer Profile','Revenue Exposure','Recommended Actions','Model & Limitations'])
    for tab,key in zip(tabs,['profile','revenue','followup','evidence']):
        with tab: st.markdown(sections[key],unsafe_allow_html=True)
    st.subheader('Download Assessment Report')
    st.caption('The HTML report includes every section, opens without an internet connection, and supports browser Print → Save as PDF.')
    a,b=st.columns(2)
    a.download_button('Download full report (HTML)',downloadable_report(customer,result,meta,sections).encode('utf-8'),file_name=f'{meta["report_id"]}_customer_report.html',mime='text/html',width='stretch')
    b.download_button('Download prediction data (JSON)',json.dumps({'report':meta,'customer':customer,'prediction':result},indent=2),file_name=f'{meta["report_id"]}_prediction.json',mime='application/json',width='stretch')
    with st.expander('Inspect submitted data and API response'):
        st.json({'report':meta,'customer':customer,'prediction':result})

with st.sidebar:
    st.markdown('<div class="brand"><span>◈</span> Churn Analytics</div><div class="small">CUSTOMER CHURN & REVENUE RISK</div>',unsafe_allow_html=True)
    st.divider()
    page=st.radio('Workspace',['Portfolio Overview','Customer Risk Assessment','Model Performance'],label_visibility='collapsed')
    st.divider()
    st.caption('PROJECT DETAILS')
    st.markdown('**Telco customer analytics**')
    st.caption('Gradient Boosting · AWS deployment')
    st.caption('Portfolio demonstration • Historical dataset')
    with st.expander('API Connection'):
        st.code(API,language=None)
        if st.button('Check API connection'):
            try:
                with urlopen(API+'/health',timeout=35) as response: health=json.load(response)
                if health.get('model_loaded'): st.success('API reachable · model loaded')
                else: st.warning('API reachable; model status unconfirmed')
            except Exception: st.error('API unreachable. Check connection and endpoint.')

if page=='Portfolio Overview':
    hero('CUSTOMER ANALYTICS','Customer Portfolio Overview','Analyze historical churn patterns, customer segments, and model-based revenue exposure across the customer portfolio.')
    df=read_csv('data/processed/telco_churn_clean.csv')
    if df is None:
        st.info('Place this dashboard.py in your project root. Overview reads data/processed/telco_churn_clean.csv. No sample figures are substituted.')
    elif not {'Churn','Contract','tenure','MonthlyCharges','customerID'}.issubset(df.columns):
        st.error('Customer CSV is missing required columns.')
    else:
        df['MonthlyCharges']=pd.to_numeric(df['MonthlyCharges'],errors='coerce')
        df['tenure']=pd.to_numeric(df['tenure'],errors='coerce')
        if df[['MonthlyCharges','tenure']].isna().any().any():
            st.error('Customer CSV has invalid numeric values. Correct the source file first.'); st.stop()
        controls=st.columns([2,2,3])
        contracts=controls[0].multiselect('Contract filter',sorted(df.Contract.unique()),default=sorted(df.Contract.unique()))
        status=controls[1].selectbox('Observed churn',['All customers','Churned','Stayed'])
        filtered=df[df.Contract.isin(contracts)].copy()
        if status!='All customers': filtered=filtered[filtered.Churn==('Yes' if status=='Churned' else 'No')]
        controls[2].caption('SOURCE: LOCAL CLEANED CUSTOMER DATA\n\nFilters apply to all overview charts and tables.')
        if filtered.empty: st.info('No customers match these filters.'); st.stop()
        a,b,c,d=st.columns(4)
        a.metric('Customers in view',f'{len(filtered):,}')
        b.metric('Observed churn rate',f'{filtered.Churn.eq("Yes").mean():.1%}')
        c.metric('Average monthly charges',f'{filtered.MonthlyCharges.mean():,.2f}')
        d.metric('Average tenure',f'{filtered.tenure.mean():.1f} months')
        st.write('')
        l,r=st.columns([1.5,1])
        with l:
            st.subheader('Churn Rate by Contract Type')
            rates=filtered.assign(churned=filtered.Churn.eq('Yes')).groupby('Contract',as_index=False).agg(Rate=('churned','mean'),Customers=('customerID','size'))
            rates['Rate']=rates.Rate*100
            fig=px.bar(rates,x='Contract',y='Rate',text_auto='.1f',hover_data=['Customers'],color_discrete_sequence=['#a78bfa'],labels={'Rate':'Observed churn (%)'})
            fig.update_traces(marker_cornerradius=7)
            chart(fig)
        with r:
            st.subheader('Customer Churn Distribution')
            counts=filtered.Churn.map({'Yes':'Churned','No':'Stayed'}).value_counts().reset_index()
            counts.columns=['Outcome','Customers']
            fig=px.pie(counts,names='Outcome',values='Customers',hole=.73,color='Outcome',color_discrete_map={'Stayed':'#a78bfa','Churned':'#67e8f9'})
            fig.update_traces(textinfo='percent',textposition='outside')
            chart(fig)
        st.subheader('Customer Retention Priorities')
        predictions=read_csv('reports/tables/batch_predictions_demo.csv')
        required={'customerID','ChurnProbability','RiskFlag','RevenueExposureScore'}
        if predictions is not None and required.issubset(predictions.columns):
            selected=predictions[predictions.customerID.isin(filtered.customerID)].copy()
            st.caption('Full-dataset batch demonstration, including training customers. Exposure is a prioritization score, not confirmed loss.')
            if selected.empty: st.info('No batch predictions match the selected customers.')
            else:
                p1,p2,p3=st.columns(3)
                p1.metric('Customers scored',f'{len(selected):,}')
                p2.metric('Flagged for review',f'{selected.RiskFlag.eq("Flagged").sum():,}')
                p3.metric('Illustrative exposure',f'{selected.RevenueExposureScore.sum():,.2f}')
                ordered=selected.sort_values('RevenueExposureScore',ascending=False)
                st.dataframe(ordered.head(25),hide_index=True,width='stretch',column_config={'ChurnProbability':st.column_config.ProgressColumn('Churn probability',min_value=0,max_value=1,format='%.3f'),'RevenueExposureScore':st.column_config.NumberColumn('Exposure score',format='%.2f')})
                st.download_button('Download filtered scores',ordered.to_csv(index=False).encode(),file_name='filtered_customer_scores.csv',mime='text/csv')
        else: st.info('Batch predictions appear here when reports/tables/batch_predictions_demo.csv is available.')

elif page=='Customer Risk Assessment':
    hero('CHURN PREDICTION','Customer Risk Assessment','Evaluate a customer profile to estimate churn probability, assess illustrative revenue exposure, and support retention planning.')
    st.caption('Defaults reproduce DEMO-001. Predictions are requested only when you submit the form.')
    service_labels={'OnlineSecurity':'Online security','OnlineBackup':'Online backup','DeviceProtection':'Device protection','TechSupport':'Tech support','StreamingTV':'Streaming TV','StreamingMovies':'Streaming movies'}
    with st.form('customer'):
        a,b,c=st.columns(3)
        with a:
            st.subheader('Customer Information')
            cid=st.text_input('Customer ID','DEMO-001')
            gender=st.selectbox('Gender',['Female','Male'])
            senior=st.selectbox('Senior citizen',['No','Yes'])
            partner=st.selectbox('Partner',['Yes','No'])
            dependents=st.selectbox('Dependents',['No','Yes'])
            tenure=st.number_input('Tenure in months',min_value=0,value=1,step=1)
        with b:
            st.subheader('Billing & Contract Details')
            contract=st.selectbox('Contract',['Month-to-month','One year','Two year'])
            payment=st.selectbox('Payment method',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
            paperless=st.selectbox('Paperless billing',['Yes','No'])
            monthly=st.number_input('Monthly charges',min_value=0.,value=29.85,step=.05)
            total=st.number_input('Total charges',min_value=0.,value=29.85,step=.05)
            st.caption('Charges use the source dataset’s monetary units.')
        with c:
            st.subheader('Subscribed Services')
            phone=st.selectbox('Phone service',['No','Yes'])
            multiple=st.checkbox('Multiple phone lines',value=False)
            internet=st.selectbox('Internet service',['DSL','Fiber optic','No'])
            addons=st.multiselect('Subscribed internet add-ons',list(service_labels.values()),default=['Online backup'])
            st.caption('Add-ons are ignored when internet service is No. Multiple lines are ignored when phone service is No.')
        submitted=st.form_submit_button('Generate Risk Assessment',type='primary')
    if submitted:
        st.session_state.pop('prediction',None)
        if not cid.strip(): st.error('Enter a customer ID.'); st.stop()
        payload=dict(customerID=cid.strip(),gender=gender,SeniorCitizen=int(senior=='Yes'),Partner=partner,Dependents=dependents,tenure=int(tenure),PhoneService=phone,MultipleLines=('No phone service' if phone=='No' else ('Yes' if multiple else 'No')),InternetService=internet,Contract=contract,PaymentMethod=payment,PaperlessBilling=paperless,MonthlyCharges=float(monthly),TotalCharges=float(total))
        payload.update({key:('No internet service' if internet=='No' else ('Yes' if label in addons else 'No')) for key,label in service_labels.items()})
        try:
            with st.spinner('Analyzing customer profile…'):
                req=Request(API+'/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
                with urlopen(req,timeout=40) as response: result=json.load(response)
            validate_prediction(payload, result)
            st.session_state.prediction=(payload,result)
            st.session_state.report_meta={
                'report_id': 'RS-' + uuid4().hex[:10].upper(),
                'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            }
        except HTTPError as exc:
            st.error('Request limit reached. Wait a few seconds and try again.' if exc.code==429 else f'Prediction failed (HTTP {exc.code}).')
            with st.expander('Response details'): st.text(exc.read().decode(errors='replace'))
        except (URLError,TimeoutError): st.error('API connection timed out or is unavailable. Please retry.')
        except (ValueError,KeyError,TypeError): st.error('Unexpected API response. Check the deployed endpoint.')
    if 'prediction' in st.session_state:
        payload, result = st.session_state.prediction
        if 'report_meta' not in st.session_state:
            st.session_state.report_meta = {
                'report_id': 'RS-' + uuid4().hex[:10].upper(),
                'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            }
        render_customer_report(payload, result, st.session_state.report_meta)

else:
    hero('MODEL EVALUATION','Model Performance & Explainability','Review cross-validation results, held-out test performance, and SHAP explanations to understand model behavior and limitations.')
    st.subheader('Held-Out Test Results')
    st.caption('Recorded project evaluation • 1,409 held-out customers • threshold 0.35. These are documented results, not live monitoring.')
    a,b,c,d=st.columns(4)
    a.metric('Test ROC-AUC','0.8457'); b.metric('Recall','71.66%'); c.metric('Precision','56.90%'); d.metric('F1 score','0.6343')
    st.write('')
    l,r=st.columns([1.4,1])
    with l:
        st.subheader('Cross-Validation Comparison')
        comparison=read_csv('reports/tables/model_comparison.csv')
        if comparison is not None and {'Model','roc_auc'}.issubset(comparison.columns):
            fig=px.bar(comparison,x='roc_auc',y='Model',orientation='h',text_auto='.4f',color_discrete_sequence=['#a78bfa'],labels={'roc_auc':'Mean validation ROC-AUC'})
            fig.update_xaxes(range=[0,1]); chart(fig)
            st.caption('Five-fold training cross-validation before hyperparameter tuning.')
        else: st.info('Model comparison chart needs reports/tables/model_comparison.csv.')
    with r:
        st.subheader('Test Set Confusion Matrix')
        fig=px.imshow([[832,203],[106,268]],x=['Predicted stayed','Predicted churn'],y=['Actually stayed','Actually churn'],text_auto=True,color_continuous_scale=['#1b1630','#a78bfa'],aspect='auto')
        fig.update_layout(coloraxis_showscale=False); chart(fig)
    st.subheader('Feature Importance & SHAP Explanations')
    images=sorted((ROOT/'reports'/'figures').glob('*shap*.png'))
    if images:
        chosen=st.selectbox('Explanation artifact',images,format_func=lambda p:p.stem.replace('_',' ').title())
        st.image(str(chosen),width='stretch')
    else: st.info('No SHAP PNGs found with “shap” in their filename under reports/figures/.')
    st.caption('SHAP describes model contributions, not causes. Contributions in this project are in log-odds, not percentage points.')
    with st.expander('Evaluation notes'):
        st.write('Training: 5,634 customers. Test: 1,409. Initial EDA used the full dataset. Hyperparameters and threshold reused the training data; cross-validation was not nested. Probability calibration and retention ROI have not been established.')

st.markdown('<div class="footer">CHURN ANALYTICS &nbsp; / &nbsp; Customer Churn & Revenue Risk Analytics &nbsp; · &nbsp; Portfolio demonstration</div>',unsafe_allow_html=True)
