import streamlit as st
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Price Prediction — LSTM Neural Network")
st.markdown("Predict stock prices using deep learning. For **educational purposes only**.")
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Settings")
ticker   = st.sidebar.text_input("Stock Ticker", "AAPL")
period   = st.sidebar.selectbox("Period", ["2y","3y","5y","10y"], index=2)
seq_len  = st.sidebar.slider("Sequence Length", 30, 90, 60)
epochs   = st.sidebar.slider("Epochs", 10, 80, 40)
run_btn  = st.sidebar.button("🚀 Run", type="primary", use_container_width=True)
st.sidebar.caption("⚠️ Not financial advice.")

if not run_btn:
    st.info("👈 Configure settings in the sidebar and click Run to begin.")
    st.stop()

# Download
with st.spinner("Downloading data..."):
    df = yf.Ticker(ticker).history(period=period).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
st.success(f"✅ {len(df)} trading days for {ticker}")

with st.expander("View raw data"):
    st.dataframe(df[["Date","Open","High","Low","Close","Volume"]].tail(10))

# Price chart
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(df["Date"],df["Close"],color="#2E75B6",linewidth=1.4)
ax.fill_between(df["Date"],df["Close"],alpha=0.1,color="#2E75B6")
ax.set_title(f"{ticker} Price History",fontweight="bold")
ax.grid(True,alpha=0.3); plt.tight_layout()
st.pyplot(fig)

# Preprocess
prices = df[["Close"]].values
sc = MinMaxScaler(); scaled = sc.fit_transform(prices)
X,y=[],[]
for i in range(seq_len,len(scaled)):
    X.append(scaled[i-seq_len:i,0]); y.append(scaled[i,0])
X=np.array(X).reshape(-1,seq_len,1); y=np.array(y)
split=int(len(X)*0.8)
X_tr,X_te,y_tr,y_te=X[:split],X[split:],y[:split],y[split:]
st.info(f"Training: {len(X_tr)} samples | Testing: {len(X_te)} samples")

# Build model
m=Sequential([LSTM(128,return_sequences=True,input_shape=(seq_len,1)),
              Dropout(0.2),LSTM(64,return_sequences=True),Dropout(0.2),
              LSTM(32,return_sequences=False),Dropout(0.2),Dense(1)])
m.compile(optimizer=Adam(0.001),loss="mse",metrics=["mae"])

# Train
st.subheader("🧠 Training")
prog=st.progress(0); stat=st.empty()
from tensorflow.keras.callbacks import LambdaCallback
cb=LambdaCallback(on_epoch_end=lambda ep,logs:
    (prog.progress(int((ep+1)/epochs*100)),
     stat.text(f"Epoch {ep+1}/{epochs} — loss: {logs['loss']:.5f}")))
hist=m.fit(X_tr,y_tr,epochs=epochs,batch_size=32,validation_data=(X_te,y_te),
           callbacks=[cb,EarlyStopping(patience=8,restore_best_weights=True)],verbose=0)
stat.success("✅ Training complete!")

# Predict
ps=m.predict(X_te,verbose=0)
pred=sc.inverse_transform(ps); actual=sc.inverse_transform(y_te.reshape(-1,1))
rmse=np.sqrt(mean_squared_error(actual,pred))
mae=mean_absolute_error(actual,pred)
mape=float(np.mean(np.abs((actual-pred)/actual))*100)
r2=r2_score(actual,pred)

# Metrics
st.subheader("📊 Results")
c1,c2,c3,c4=st.columns(4)
c1.metric("RMSE",f"${rmse:.2f}")
c2.metric("MAE",f"${mae:.2f}")
c3.metric("MAPE",f"{mape:.2f}%")
c4.metric("R²",f"{r2:.4f}")

# Chart
fig2,ax2=plt.subplots(figsize=(12,5))
ax2.plot(actual,color="#1F4E79",lw=2,label="Actual")
ax2.plot(pred,color="#C00000",lw=1.5,ls="--",label="Predicted")
ax2.set_title(f"{ticker}: Actual vs Predicted",fontweight="bold")
ax2.legend(); ax2.grid(True,alpha=0.3); plt.tight_layout()
st.pyplot(fig2)

# Next day
nxt=sc.inverse_transform(m.predict(scaled[-seq_len:].reshape(1,seq_len,1),verbose=0))[0][0]
today=df["Close"].iloc[-1]; delta=nxt-today; pct=delta/today*100
st.subheader("🔮 Next Day Forecast")
col1,col2=st.columns(2)
col1.metric("Today Close",f"${today:.2f}")
col2.metric("Predicted Next",f"${nxt:.2f}",f"{'↑' if delta>0 else '↓'} {pct:+.2f}%")

# Download
csv=pd.DataFrame({"Actual":actual.flatten(),"Predicted":pred.flatten()})
st.download_button("⬇️ Download CSV",csv.to_csv(index=False),f"{ticker}_results.csv")
st.caption("Educational use only — not financial advice.")