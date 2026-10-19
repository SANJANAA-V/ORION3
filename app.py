"""ORION3 demo app (Streamlit). Run locally:  streamlit run app.py
On Hugging Face Spaces choose the Streamlit SDK and upload: app.py, orion3_core.py, orion3_sim_v3.py, requirements.txt.
The models are trained on first start (about 30 seconds) and then reused."""
import streamlit as st
import orion3_core as core

st.set_page_config(page_title="ORION³ — scam-risk demo", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="First start: training the models on simulated data (about 30 seconds)...")
def get_bundle():
    return core.load_bundle()


try:
    bundle = get_bundle()
except Exception as e:           # show the real problem instead of a blank page
    st.error(f"The demo could not start: {type(e).__name__}: {e}")
    st.stop()

st.title("ORION³ — pre-payment scam-risk demo")
st.warning("Research demo: payment behaviour is SIMULATED, the message model is trained on real public data. It is not connected to any bank or UPI system "
           "and must not be used for real payment decisions.")

preset_name = st.selectbox("Start from an example", list(core.PRESETS.keys()))
P = core.PRESETS[preset_name]
k = preset_name  # widget keys change with the example, so the inputs reset when you pick another one

st.subheader("The payment")
c1, c2 = st.columns(2)
amount = c1.number_input("Amount (₹)", min_value=0.0, value=P["amount"], step=100.0, key="a" + k)
typical = c2.number_input("Sender's usual payment (₹)", min_value=1.0, value=P["typical_amount"], step=50.0, key="t" + k)
n_prior = c1.number_input("Sender's earlier payments (count)", min_value=0, value=int(P["n_prior_payments"]), key="n" + k)
hour = c2.slider("Hour of day", 0, 23, int(P["hour"]), key="h" + k)
new_rcv = c1.checkbox("First payment to this receiver", value=P["new_receiver"], key="nr" + k)
merchant = c2.checkbox("Receiver is a registered merchant", value=P["receiver_is_merchant"], key="m" + k)
message = st.text_area("Message that led to this payment (leave empty if none)", value=P["message"], key="msg" + k)

st.subheader("What we know about the receiving account (last 60 minutes)")
d1, d2, d3 = st.columns(3)
uniq = d1.number_input("Different people who paid it", min_value=0, value=int(P["rcv_unique_senders_60m"]), key="u" + k)
txns = d2.number_input("Payments it received", min_value=0, value=int(P["rcv_txns_60m"]), key="x" + k)
out = d3.number_input("Payments it sent onward", min_value=0, value=int(P["rcv_out_60m"]), key="o" + k)
age = st.number_input("Account age in minutes (-1 = never seen before)", min_value=-1, value=int(P["rcv_age_min"]), key="ag" + k)

if st.button("Assess this payment", type="primary"):
    r = core.score_payment(bundle, dict(
        amount=amount, typical_amount=typical, n_prior_payments=n_prior, new_receiver=new_rcv,
        rcv_unique_senders_60m=uniq, rcv_txns_60m=txns, rcv_out_60m=out, rcv_age_min=age,
        hour=hour, receiver_is_merchant=merchant, message=message))
    banner = {"ALLOW": st.success, "WARN": st.warning, "BLOCK": st.error}[r["action"]]
    banner(f"**{r['action']}** — risk score {r['risk_score']:.3f}")
    st.markdown("**Why**")
    for reason in r["reasons"]:
        st.markdown(f"- {reason}")
    m1, m2 = st.columns(2)
    s = r["signals"]
    m1.metric("Behaviour + receiver network", f"{s['behaviour_and_network']:.3f}")
    m2.metric("Message", "no message" if s["message"] is None else f"{s['message']:.3f}")
    st.caption(f"Warn at risk ≥ {r['thresholds']['warn']:.3f}, block at ≥ {r['thresholds']['block']:.3f} "
               f"(set on simulated validation data). Scored in {r['latency_ms']:.1f} ms. "
               "The risk score is a model output, not a calibrated probability.")

with st.expander("What is real and what is simulated?"):
    st.markdown(
        "- **Simulated:** all payments, accounts, scam campaigns and messages used to train these models.\n"
        "- **Real:** the message model is trained on real public data (real smishing messages from the IMC 2025 dataset and real "
        "legitimate SMS). No real payment data is used. Benchmarks on real fraud data (IEEE-CIS) are in the GitHub repository.\n"
        "- The message model is untested on real legitimate bank and UPI alerts, and it adds little on top of the payment signals in my tests.\n"
        "- Reasons come from the fusion model's own weights, not from a separate text generator.")
