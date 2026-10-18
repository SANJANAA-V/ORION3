"""ORION3 scoring API.   Run:  uvicorn api:app --port 8000
Open http://localhost:8000/docs to try it in the browser."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
import orion3_core as core

app = FastAPI(title="ORION3 pre-payment scam-risk API", version="0.1",
              description="Research prototype trained on SIMULATED data. Not for real payment decisions.")
bundle = core.load_bundle()        # trains once at start-up if no saved models exist


class Payment(BaseModel):
    amount: float = Field(..., ge=0, description="Payment amount in rupees")
    typical_amount: float = Field(400.0, gt=0, description="Sender's usual payment")
    n_prior_payments: int = Field(0, ge=0, description="How many payments the sender made before")
    new_receiver: bool = Field(False, description="First payment to this receiver")
    rcv_unique_senders_60m: int = Field(0, ge=0, description="Different people who paid the receiver in the last 60 min")
    rcv_txns_60m: int = Field(0, ge=0, description="Payments the receiver got in the last 60 min")
    rcv_out_60m: int = Field(0, ge=0, description="Payments the receiver sent onward in the last 60 min")
    rcv_age_min: int = Field(-1, ge=-1, description="Receiver account age in minutes (-1 = never seen)")
    hour: int = Field(12, ge=0, le=23)
    receiver_is_merchant: bool = False
    message: str = Field("", description="Message that led to the payment (empty if none)")


@app.get("/health")
def health():
    return {"status": "ok", "meta": bundle["meta"]}


@app.post("/score")
def score(p: Payment):
    f = p.model_dump() if hasattr(p, "model_dump") else p.dict()
    return core.score_payment(bundle, f)
